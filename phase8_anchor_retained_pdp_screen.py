from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import r2_pipeline as rp
from phase8_anchor_retained_pas_screen import farthest_subset, mapped_anchors


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
K_VALUES = (1, 2, 4, 8, 99)
POWERS = (0.0, 1.0, 2.0)
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
MODES = ("direct", "residual", "ratio")
EPSILON = 1e-3 / 24


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


@torch.no_grad()
def pdp_descriptor(channel: np.ndarray) -> np.ndarray:
    rows = []
    for start in range(0, len(channel), 4):
        stop = min(start + 4, len(channel))
        x = torch.as_tensor(np.asarray(channel[start:stop]).copy(), device=DEVICE)
        pdp = torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2
        pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
        pdp = pdp.reshape(len(x), 256, 4, 24, 8).mean(-1)
        pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
        rows.append(pdp.cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def descriptors(path: Path, truth: np.ndarray, cache_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    base_path = ROOT / f"{cache_prefix}_base_pdp_band24.npy"
    truth_path = ROOT / f"{cache_prefix}_truth_pdp_band24.npy"
    if base_path.exists() and truth_path.exists():
        base = np.load(base_path)
        target = np.load(truth_path)
        if base.shape[-1] == 24 and target.shape[-1] == 24:
            return base, target
    base = pdp_descriptor(np.load(path, mmap_mode="r"))
    target = pdp_descriptor(truth)
    np.save(base_path, base)
    np.save(truth_path, target)
    return base, target


def predict_anchor_descriptors(
    xy: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    query: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    k_value: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_truth = base[query].copy()
    predicted_base = base[query].copy()
    predicted_log_ratio = np.zeros_like(predicted_truth)
    for group in np.unique(labels[query]):
        anchor = anchors[labels[anchors] == group]
        rows = np.flatnonzero(labels[query] == group)
        if not len(anchor):
            continue
        k = min(k_value, len(anchor))
        distance, local = cKDTree(xy[anchor]).query(xy[query[rows]], k=k)
        distance = np.asarray(distance)
        local = np.asarray(local)
        if distance.ndim == 1:
            distance = distance[:, None]
            local = local[:, None]
        if power == 0:
            weight = np.ones_like(distance, dtype=np.float64)
        else:
            scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
            weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** power
        weight /= weight.sum(1, keepdims=True)
        for offset, row in enumerate(rows):
            chosen = anchor[local[offset]]
            w = weight[offset]
            predicted_truth[row] = np.einsum(
                "k,kaus->aus", w, truth[chosen], optimize=True
            )
            predicted_base[row] = np.einsum(
                "k,kaus->aus", w, base[chosen], optimize=True
            )
            predicted_log_ratio[row] = np.einsum(
                "k,kaus->aus",
                w,
                np.log((truth[chosen] + EPSILON) / (base[chosen] + EPSILON)),
                optimize=True,
            )
    return predicted_truth, predicted_base, predicted_log_ratio


def weighted_c2(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    numerator = np.sum(prediction * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    cosine = numerator / np.maximum(denominator, 1e-30)
    return float(np.sum(cosine * weights[:, None, None]) / (weights.sum() * 256 * 4))


def official_geometry(pos: np.ndarray, energy: np.ndarray, test_pos: np.ndarray):
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    official_counts = {
        int(group): int(count)
        for group, count in zip(*np.unique(test_labels, return_counts=True))
    }
    boxes = {
        group: (
            test_pos[test_labels == group, :2].min(0),
            test_pos[test_labels == group, :2].max(0),
        )
        for group in official_counts
    }
    train_labels = np.full(len(pos), -1, dtype=np.int64)
    for index in np.flatnonzero(energy > 0):
        for group in sorted(official_counts):
            lo, hi = boxes[group]
            if np.all(pos[index, :2] >= lo) and np.all(pos[index, :2] <= hi):
                train_labels[index] = group
                break
    fractions = {}
    counts = {}
    for group in official_counts:
        lo, hi = boxes[group]
        anchor = pos[train_labels == group, :2]
        fractions[group] = (anchor - lo) / np.maximum(hi - lo, 1e-9)
        counts[group] = int(len(anchor))
    return official_counts, fractions, counts


def run() -> None:
    pos, channel, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    official_counts, actual_fraction, actual_counts = official_geometry(
        pos, energy, test_pos
    )
    keys = list(itertools.product(K_VALUES, POWERS, MODES, ALPHAS))
    fold_values = {key: [] for key in keys}
    fold_baselines = []
    anchor_counts = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        query = np.setdiff1d(np.arange(len(val)), anchors)
        base, truth = descriptors(
            ROOT / f"matched_phase6_full_fold{fold}.npy",
            channel[val],
            f"phase8_anchor_retained_fold{fold}",
        )
        weights = np.asarray(
            [official_counts[int(group)] / np.sum(labels[query] == group) for group in labels[query]],
            dtype=np.float64,
        )
        baseline = weighted_c2(base[query], truth[query], weights)
        fold_baselines.append(baseline)
        anchor_counts.append(
            {str(group): int(np.sum(labels[anchors] == group)) for group in official_counts}
        )
        xy = pos[val, :2]
        for k_value, power in itertools.product(K_VALUES, POWERS):
            anchor_truth, anchor_base, anchor_log_ratio = predict_anchor_descriptors(
                xy, labels, anchors, query, base, truth, k_value, power
            )
            for mode, alpha in itertools.product(MODES, ALPHAS):
                if mode == "direct":
                    desired = normalize((1.0 - alpha) * base[query] + alpha * anchor_truth)
                elif mode == "residual":
                    desired = normalize(base[query] + alpha * (anchor_truth - anchor_base))
                else:
                    desired = normalize(base[query] * np.exp(alpha * anchor_log_ratio))
                fold_values[(k_value, power, mode, alpha)].append(
                    weighted_c2(desired, truth[query], weights)
                )
        print(
            json.dumps(
                {
                    "stage": "fold",
                    "fold": fold,
                    "baseline_c2_band24": baseline,
                    "anchors": anchor_counts[-1],
                    "queries": int(len(query)),
                }
            ),
            flush=True,
        )
    baseline_array = np.asarray(fold_baselines)
    rows = []
    for key in keys:
        values = np.asarray(fold_values[key])
        delta = 0.4 * (values - baseline_array)
        rows.append(
            {
                "k": key[0],
                "power": key[1],
                "mode": key[2],
                "alpha": key[3],
                "c2_band24": values.tolist(),
                "score_proxy_deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - delta.std()),
            }
        )
    rows.sort(key=lambda row: (row["lcb"], row["mean_delta"]), reverse=True)
    output = {
        "official_test_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "fold_anchor_counts": anchor_counts,
        "fold_baseline_c2_band24": fold_baselines,
        "results": rows,
    }
    (ROOT / "phase8_anchor_retained_pdp_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "top": rows[:20]}), flush=True)


if __name__ == "__main__":
    run()
