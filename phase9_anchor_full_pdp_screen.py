from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import r2_pipeline as rp
from phase8_anchor_retained_pas_screen import mapped_anchors
from phase8_anchor_retained_pdp_screen import official_geometry


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
NEIGHBORS = (4, 8, 16)
POWERS = (0.5, 1.0, 2.0)
ALPHAS = (0.025, 0.05, 0.10, 0.15, 0.20)


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


@torch.no_grad()
def descriptor(channel: np.ndarray, path: Path) -> np.ndarray:
    if path.exists():
        return np.load(path, mmap_mode="r")
    rows = []
    for start in range(0, len(channel), 4):
        stop = min(start + 4, len(channel))
        x = torch.as_tensor(np.asarray(channel[start:stop]).copy(), device=DEVICE)
        pdp = torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2
        pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
        rows.append(pdp.cpu().numpy().astype(np.float32))
    output = np.concatenate(rows)
    np.save(path, output)
    return np.load(path, mmap_mode="r")


def run() -> None:
    pos, channel, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    official_counts, actual_fraction, actual_counts = official_geometry(pos, energy, test_pos)
    keys = list(itertools.product(NEIGHBORS, POWERS, ALPHAS))
    fold_scores = {key: [] for key in keys}
    baselines = []
    epsilon = 1e-4 / 192
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        query_local = np.setdiff1d(np.arange(len(val)), anchors)
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        base_all = descriptor(
            prediction,
            ROOT / f"phase9_anchor_fold{fold}_base_pdp_band192.npy",
        )
        truth_all = descriptor(
            channel[val],
            ROOT / f"phase9_anchor_fold{fold}_truth_pdp_band192.npy",
        )
        weights = np.asarray(
            [official_counts[int(group)] / np.sum(labels[query_local] == group)
             for group in labels[query_local]], dtype=np.float64
        )
        denominator = float(weights.sum()) * 256 * 4
        baseline_sum = 0.0
        accumulator = {key: 0.0 for key in keys}
        for group in np.unique(labels[query_local]):
            rows = np.flatnonzero(labels[query_local] == group)
            local_query = query_local[rows]
            group_anchors = anchors[labels[anchors] == group]
            base = np.asarray(base_all[local_query])
            truth = np.asarray(truth_all[local_query])
            base_cos = np.sum(base * truth, axis=-1) / np.maximum(
                np.linalg.norm(base, axis=-1) * np.linalg.norm(truth, axis=-1), 1e-30
            )
            weighted_base = float(np.sum(base_cos * weights[rows, None, None]))
            baseline_sum += weighted_base
            if len(group_anchors) == 0:
                for key in keys:
                    accumulator[key] += weighted_base
                continue
            k_max = len(group_anchors)
            distance, local = cKDTree(pos[val[group_anchors], :2]).query(
                pos[val[local_query], :2], k=k_max
            )
            distance = np.asarray(distance)
            local = np.asarray(local)
            if k_max == 1:
                distance = distance[:, None]
                local = local[:, None]
            anchor_truth = np.asarray(truth_all[group_anchors])
            anchor_base = np.asarray(base_all[group_anchors])
            anchor_log = np.clip(
                np.log((anchor_truth + epsilon) / (anchor_base + epsilon)), -2.0, 2.0
            ).astype(np.float32)
            selected = anchor_log[local]
            for neighbors, power in itertools.product(NEIGHBORS, POWERS):
                k = min(neighbors, k_max)
                scale = np.maximum(np.median(distance[:, :k], axis=1, keepdims=True), 1.0)
                neighbor_weight = 1.0 / np.maximum(
                    distance[:, :k] + 0.10 * scale, 0.25
                ) ** power
                neighbor_weight /= neighbor_weight.sum(1, keepdims=True)
                predicted_log = np.einsum(
                    "rk,rkaus->raus", neighbor_weight, selected[:, :k], optimize=True
                )
                for alpha in ALPHAS:
                    desired = normalize(base * np.exp(alpha * predicted_log))
                    cosine = np.sum(desired * truth, axis=-1) / np.maximum(
                        np.linalg.norm(desired, axis=-1) * np.linalg.norm(truth, axis=-1),
                        1e-30,
                    )
                    accumulator[neighbors, power, alpha] += float(
                        np.sum(cosine * weights[rows, None, None])
                    )
        baseline = baseline_sum / denominator
        baselines.append(baseline)
        for key in keys:
            fold_scores[key].append(accumulator[key] / denominator)
        print(json.dumps({"stage": "fold", "fold": fold, "baseline_c2": baseline}), flush=True)

    baseline_array = np.asarray(baselines)
    rows = []
    for key in keys:
        delta = 0.4 * (np.asarray(fold_scores[key]) - baseline_array)
        rows.append(
            {
                "neighbors": key[0], "power": key[1], "alpha": key[2],
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
        "fold_baseline_c2": baselines,
        "results": rows,
    }
    (ROOT / "phase9_anchor_full_pdp_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "top": rows[:30]}), flush=True)


if __name__ == "__main__":
    run()
