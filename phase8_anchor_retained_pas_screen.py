from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import base_descriptor


ROOT = Path(__file__).resolve().parent
K_VALUES = (1, 2, 4, 8, 99)
POWERS = (0.0, 1.0, 2.0)
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
MODES = ("direct", "residual", "ratio")
ALIGNMENTS = ("none", "horizontal")
EPSILON = 1e-3 / 24


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def horizontal_shifts(pos: np.ndarray) -> np.ndarray:
    xy = pos[:, :2]
    side = xy[:, 1] > 0
    bs = np.where(
        side[:, None],
        np.array([52.0, 35.0]),
        np.array([-18.413, -65.881]),
    )
    relative = xy - bs
    unit = relative / np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-9)
    return np.rint(5.0 * unit[:, 1]).astype(np.int64)


def roll_horizontal(value: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    shaped = value.reshape(len(value), 2, 16, 8, 4, 24)
    output = np.empty_like(shaped)
    for row, shift in enumerate(shifts):
        output[row] = np.roll(shaped[row], int(shift), axis=1)
    return output.reshape(value.shape)


def farthest_subset(points: np.ndarray, count: int) -> np.ndarray:
    if count >= len(points):
        return np.arange(len(points), dtype=np.int64)
    center = points.mean(0)
    chosen = [int(np.argmin(np.sum((points - center) ** 2, axis=1)))]
    distance = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    while len(chosen) < count:
        distance[chosen] = -1
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(
            distance, np.sum((points - points[index]) ** 2, axis=1)
        )
    return np.asarray(chosen, dtype=np.int64)


def mapped_anchors(
    pos: np.ndarray,
    val: np.ndarray,
    labels: np.ndarray,
    actual_fraction: dict[int, np.ndarray],
    official_test_counts: dict[int, int],
) -> np.ndarray:
    selected: list[int] = []
    for group in sorted(official_test_counts):
        local = np.flatnonzero(labels == group)
        fractions = actual_fraction[group]
        if not len(local) or not len(fractions):
            continue
        fraction = len(fractions) / (official_test_counts[group] + len(fractions))
        count = int(np.rint(len(local) * fraction))
        count = min(max(count, 1), len(local), len(fractions))
        template = fractions[farthest_subset(fractions, count)]
        xy = pos[val[local], :2]
        lo, hi = xy.min(0), xy.max(0)
        mapped = lo + template * np.maximum(hi - lo, 1e-9)
        available = np.ones(len(local), dtype=bool)
        for point in mapped:
            distance = np.sum((xy - point) ** 2, axis=1)
            distance[~available] = np.inf
            pick = int(np.argmin(distance))
            selected.append(int(local[pick]))
            available[pick] = False
    return np.asarray(sorted(selected), dtype=np.int64)


def descriptors(path: Path, truth: np.ndarray, cache_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    base_path = ROOT / f"{cache_prefix}_base_pas_band24.npy"
    truth_path = ROOT / f"{cache_prefix}_truth_pas_band24.npy"
    if base_path.exists() and truth_path.exists():
        base = np.load(base_path)
        target = np.load(truth_path)
        if base.shape[-1] == 24 and target.shape[-1] == 24:
            return base, target
    base = base_descriptor(np.load(path, mmap_mode="r")).astype(np.float32)
    target = base_descriptor(truth).astype(np.float32)
    np.save(base_path, base)
    np.save(truth_path, target)
    return base, target


def predict_anchor_descriptors(
    xy: np.ndarray,
    labels: np.ndarray,
    shifts: np.ndarray,
    anchors: np.ndarray,
    query: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    k_value: int,
    power: float,
    alignment: str,
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
            anchor_truth = truth[chosen]
            anchor_base = base[chosen]
            if alignment == "horizontal":
                delta = shifts[query[row]] - shifts[chosen]
                anchor_truth = roll_horizontal(anchor_truth, delta)
                anchor_base = roll_horizontal(anchor_base, delta)
            w = weight[offset]
            predicted_truth[row] = np.einsum(
                "k,kaub->aub", w, anchor_truth, optimize=True
            )
            predicted_base[row] = np.einsum(
                "k,kaub->aub", w, anchor_base, optimize=True
            )
            predicted_log_ratio[row] = np.einsum(
                "k,kaub->aub",
                w,
                np.log((anchor_truth + EPSILON) / (anchor_base + EPSILON)),
                optimize=True,
            )
    return predicted_truth, predicted_base, predicted_log_ratio


def weighted_c1(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> float:
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-30)
    return float(np.sum(cosine * weights[:, None, None]) / (weights.sum() * 4 * 24))


def run() -> None:
    pos, channel, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
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
    # Assign overlapping rectangle edges once, using the same first-match rule
    # as the exact-box OOF audit that established the 75 available anchors.
    train_box_labels = np.full(len(pos), -1, dtype=np.int64)
    for index in np.flatnonzero(energy > 0):
        for group in sorted(official_counts):
            lo, hi = boxes[group]
            if np.all(pos[index, :2] >= lo) and np.all(pos[index, :2] <= hi):
                train_box_labels[index] = group
                break
    actual_fraction: dict[int, np.ndarray] = {}
    actual_counts: dict[int, int] = {}
    for group in official_counts:
        lo, hi = boxes[group]
        anchor = pos[train_box_labels == group, :2]
        actual_fraction[group] = (anchor - lo) / np.maximum(hi - lo, 1e-9)
        actual_counts[group] = int(len(anchor))

    keys = list(itertools.product(ALIGNMENTS, K_VALUES, POWERS, MODES, ALPHAS))
    fold_values = {key: [] for key in keys}
    fold_baselines: list[float] = []
    anchor_counts = []
    shifts_all = horizontal_shifts(pos)
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        query = np.setdiff1d(np.arange(len(val)), anchors)
        cache_prefix = f"phase8_anchor_retained_fold{fold}"
        base, truth = descriptors(
            ROOT / f"matched_phase6_full_fold{fold}.npy",
            channel[val],
            cache_prefix,
        )
        weights = np.asarray(
            [official_counts[int(group)] / np.sum(labels[query] == group) for group in labels[query]],
            dtype=np.float64,
        )
        baseline = weighted_c1(base[query], truth[query], weights)
        fold_baselines.append(baseline)
        anchor_counts.append(
            {
                str(group): int(np.sum(labels[anchors] == group))
                for group in official_counts
            }
        )
        xy = pos[val, :2]
        shifts = shifts_all[val]
        for alignment, k_value, power in itertools.product(ALIGNMENTS, K_VALUES, POWERS):
            anchor_truth, anchor_base, anchor_log_ratio = predict_anchor_descriptors(
                xy,
                labels,
                shifts,
                anchors,
                query,
                base,
                truth,
                k_value,
                power,
                alignment,
            )
            for mode, alpha in itertools.product(MODES, ALPHAS):
                if mode == "direct":
                    desired = normalize((1.0 - alpha) * base[query] + alpha * anchor_truth)
                elif mode == "residual":
                    desired = normalize(
                        base[query] + alpha * (anchor_truth - anchor_base)
                    )
                else:
                    desired = normalize(base[query] * np.exp(alpha * anchor_log_ratio))
                fold_values[(alignment, k_value, power, mode, alpha)].append(
                    weighted_c1(desired, truth[query], weights)
                )
        print(
            json.dumps(
                {
                    "stage": "fold",
                    "fold": fold,
                    "baseline_c1_band24": baseline,
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
                "alignment": key[0],
                "k": key[1],
                "power": key[2],
                "mode": key[3],
                "alpha": key[4],
                "c1_band24": values.tolist(),
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
        "fold_baseline_c1_band24": fold_baselines,
        "results": rows,
    }
    (ROOT / "phase8_anchor_retained_pas_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "top": rows[:20]}), flush=True)


if __name__ == "__main__":
    run()
