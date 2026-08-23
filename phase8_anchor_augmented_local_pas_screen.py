from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import r2_pipeline as rp
from phase8_anchor_retained_pas_screen import (
    horizontal_shifts,
    mapped_anchors,
    normalize,
    roll_horizontal,
    weighted_c1,
)
from phase8_anchor_retained_pdp_screen import official_geometry


ROOT = Path(__file__).resolve().parent
K_VALUES = (1, 2, 4, 8, 16, 32, 64)
POWERS = (0.5, 1.0, 2.0, 3.0)
ALIGNMENTS = ("none", "horizontal")
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def local_prediction(
    xy: np.ndarray,
    shifts: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    k_value: int,
    power: float,
    alignment: str,
) -> np.ndarray:
    k = min(k_value, len(train))
    distance, local = cKDTree(xy[train]).query(xy[query], k=k)
    distance = np.asarray(distance)
    local = np.asarray(local)
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
    weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** power
    weight /= weight.sum(1, keepdims=True)
    output = np.empty((len(query), 256, 4, 24), dtype=np.float32)
    indices = train[local]
    for row in range(len(query)):
        value = np.asarray(target[indices[row]])
        if alignment == "horizontal":
            value = roll_horizontal(
                value, shifts[query[row]] - shifts[indices[row]]
            )
        output[row] = np.einsum(
            "k,kaub->aub", weight[row], value, optimize=True
        )
    return output


def run() -> None:
    pos, _, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    official_counts, actual_fraction, actual_counts = official_geometry(
        pos, energy, test_pos
    )
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    shifts = horizontal_shifts(pos)
    valid = energy > 0
    keys = list(itertools.product(K_VALUES, POWERS, ALIGNMENTS, ALPHAS))
    fold_values = {key: [] for key in keys}
    fold_baselines = []
    neighbor_quantiles = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        query_local = np.setdiff1d(np.arange(len(val)), anchors)
        query = val[query_local]
        train_mask = valid.copy()
        train_mask[val] = False
        train_mask[val[anchors]] = True
        train = np.flatnonzero(train_mask)
        base = np.load(
            ROOT / f"phase8_anchor_retained_fold{fold}_base_pas_band24.npy"
        )[query_local]
        truth = np.asarray(target[query])
        weights = np.asarray(
            [
                official_counts[int(group)] / np.sum(labels[query_local] == group)
                for group in labels[query_local]
            ],
            dtype=np.float64,
        )
        baseline = weighted_c1(base, truth, weights)
        fold_baselines.append(baseline)
        nearest = cKDTree(pos[train, :2]).query(pos[query, :2], k=1)[0]
        neighbor_quantiles.append(np.quantile(nearest, [0, 0.25, 0.5, 0.75, 1]).tolist())
        for k_value, power, alignment in itertools.product(
            K_VALUES, POWERS, ALIGNMENTS
        ):
            local = local_prediction(
                pos[:, :2], shifts, target, train, query, k_value, power, alignment
            )
            for alpha in ALPHAS:
                desired = normalize((1.0 - alpha) * base + alpha * local)
                fold_values[(k_value, power, alignment, alpha)].append(
                    weighted_c1(desired, truth, weights)
                )
        print(
            json.dumps(
                {
                    "stage": "fold",
                    "fold": fold,
                    "baseline_c1_band24": baseline,
                    "anchors": int(len(anchors)),
                    "queries": int(len(query)),
                    "nearest_distance_quantiles": neighbor_quantiles[-1],
                }
            ),
            flush=True,
        )
    baseline_array = np.asarray(fold_baselines)
    rows = []
    for key in keys:
        values = np.asarray(fold_values[key])
        deltas = 0.4 * (values - baseline_array)
        rows.append(
            {
                "k": key[0],
                "power": key[1],
                "alignment": key[2],
                "alpha": key[3],
                "c1_band24": values.tolist(),
                "score_proxy_deltas": deltas.tolist(),
                "mean_delta": float(deltas.mean()),
                "min_delta": float(deltas.min()),
                "lcb": float(deltas.mean() - deltas.std()),
            }
        )
    rows.sort(key=lambda row: (row["lcb"], row["mean_delta"]), reverse=True)
    output = {
        "official_test_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "nearest_distance_quantiles": neighbor_quantiles,
        "fold_baseline_c1_band24": fold_baselines,
        "results": rows,
    }
    (ROOT / "phase8_anchor_augmented_local_pas_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "top": rows[:25]}), flush=True)


if __name__ == "__main__":
    run()
