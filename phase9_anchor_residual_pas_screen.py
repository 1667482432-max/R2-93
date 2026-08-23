from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import r2_pipeline as rp
from phase8_anchor_retained_pas_screen import mapped_anchors, normalize, weighted_c1
from phase8_anchor_retained_pdp_screen import official_geometry


ROOT = Path(__file__).resolve().parent
NEIGHBORS = (1, 2, 4, 8, 16)
POWERS = (1.0, 2.0, 3.0)
MODES = ("additive", "log_ratio")
ALPHAS = (0.025, 0.05, 0.10, 0.15, 0.20, 0.30)


def residual_family(
    pos: np.ndarray,
    val: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    query_local: np.ndarray,
    base_all: np.ndarray,
    target: np.ndarray,
) -> dict[tuple[int, float, str], np.ndarray]:
    base = base_all[query_local]
    query = val[query_local]
    keys = list(itertools.product(NEIGHBORS, POWERS, MODES))
    output = {key: np.zeros_like(base, dtype=np.float32) for key in keys}
    epsilon = 1e-4 / 256
    for group in np.unique(labels[query_local]):
        rows = np.flatnonzero(labels[query_local] == group)
        group_anchors = anchors[labels[anchors] == group]
        if len(group_anchors) == 0:
            continue
        k_max = min(max(NEIGHBORS), len(group_anchors))
        distance, local = cKDTree(pos[val[group_anchors], :2]).query(
            pos[query[rows], :2], k=k_max
        )
        distance = np.asarray(distance)
        local = np.asarray(local)
        if k_max == 1:
            distance = distance[:, None]
            local = local[:, None]
        anchor_local = group_anchors[local]
        truth_anchor = np.asarray(target[val[anchor_local]])
        base_anchor = base_all[anchor_local]
        residuals = {
            "additive": truth_anchor - base_anchor,
            "log_ratio": np.clip(
                np.log((truth_anchor + epsilon) / (base_anchor + epsilon)), -2.0, 2.0
            ),
        }
        for neighbors, power, mode in keys:
            k = min(neighbors, k_max)
            scale = np.maximum(np.median(distance[:, :k], axis=1, keepdims=True), 1.0)
            weight = 1.0 / np.maximum(distance[:, :k] + 0.10 * scale, 0.25) ** power
            weight /= weight.sum(1, keepdims=True)
            output[neighbors, power, mode][rows] = np.einsum(
                "rk,rkaub->raub", weight, residuals[mode][:, :k], optimize=True
            )
    return output


def run() -> None:
    pos, _, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    official_counts, actual_fraction, actual_counts = official_geometry(pos, energy, test_pos)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    keys = list(itertools.product(NEIGHBORS, POWERS, MODES, ALPHAS))
    values = {key: [] for key in keys}
    baselines = []
    anchor_counts = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        query_local = np.setdiff1d(np.arange(len(val)), anchors)
        query = val[query_local]
        base_all = np.load(ROOT / f"phase8_anchor_retained_fold{fold}_base_pas_band24.npy")
        base = base_all[query_local]
        truth = np.asarray(target[query])
        weights = np.asarray(
            [official_counts[int(group)] / np.sum(labels[query_local] == group)
             for group in labels[query_local]],
            dtype=np.float64,
        )
        baseline = weighted_c1(base, truth, weights)
        baselines.append(baseline)
        family = residual_family(pos, val, labels, anchors, query_local, base_all, target)
        for neighbors, power, mode, alpha in keys:
            residual = family[neighbors, power, mode]
            if mode == "additive":
                desired = normalize(np.maximum(base + alpha * residual, 0.0))
            else:
                desired = normalize(base * np.exp(alpha * residual))
            values[neighbors, power, mode, alpha].append(
                weighted_c1(desired, truth, weights)
            )
        counts = {
            str(group): int(np.sum(labels[anchors] == group))
            for group in np.unique(labels)
        }
        anchor_counts.append(counts)
        print(
            json.dumps({"stage": "fold", "fold": fold, "baseline": baseline, "anchors": counts}),
            flush=True,
        )

    baseline_array = np.asarray(baselines)
    rows = []
    for key in keys:
        score = np.asarray(values[key])
        delta = 0.4 * (score - baseline_array)
        rows.append(
            {
                "neighbors": key[0],
                "power": key[1],
                "mode": key[2],
                "alpha": key[3],
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
        "validation_anchor_counts": anchor_counts,
        "fold_baseline_c1_band24": baselines,
        "results": rows,
    }
    (ROOT / "phase9_anchor_residual_pas_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "top": rows[:30]}), flush=True)


if __name__ == "__main__":
    run()
