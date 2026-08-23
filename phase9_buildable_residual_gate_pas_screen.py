from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

from phase8_anchor_augmented_local_pas_screen import local_prediction
from phase8_anchor_local_gate_channel_validation import ALPHA_GRID, prepare_folds
from phase8_anchor_retained_pas_screen import horizontal_shifts, mapped_anchors, normalize, weighted_c1
from phase8_anchor_retained_pdp_screen import official_geometry


ROOT = Path(__file__).resolve().parent
EXTERNAL_ALIGNMENTS = ("none", "horizontal")
RESIDUAL_ALPHAS = (0.10, 0.15, 0.20, 0.25)
LOCAL_SCALES = (0.50, 0.75)
ORDERS = ("residual_then_local", "local_then_residual")


def external_residual(
    pos: np.ndarray,
    val: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    query_local: np.ndarray,
    target: np.ndarray,
    external_train: np.ndarray,
    shifts: np.ndarray,
    alignment: str,
) -> np.ndarray:
    anchor_index = val[anchors]
    anchor_base = local_prediction(
        pos[:, :2], shifts, target, external_train, anchor_index, 4, 3.0, alignment
    )
    anchor_truth = np.asarray(target[anchor_index])
    epsilon = 1e-4 / 256
    anchor_log = np.clip(
        np.log((anchor_truth + epsilon) / (anchor_base + epsilon)), -2.0, 2.0
    )
    anchor_log = np.repeat(anchor_log.mean(3, keepdims=True), anchor_log.shape[3], axis=3)
    output = np.zeros((len(query_local), 256, 4, 24), dtype=np.float32)
    for group in np.unique(labels[query_local]):
        rows = np.flatnonzero(labels[query_local] == group)
        group_anchors = np.flatnonzero(labels[anchors] == group)
        if len(group_anchors) == 0:
            continue
        k = min(16, len(group_anchors))
        distance, local = cKDTree(pos[anchor_index[group_anchors], :2]).query(
            pos[val[query_local[rows]], :2], k=k
        )
        distance = np.asarray(distance)
        local = np.asarray(local)
        if k == 1:
            distance = distance[:, None]
            local = local[:, None]
        scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
        weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25)
        weight /= weight.sum(1, keepdims=True)
        output[rows] = np.einsum(
            "rk,rkaub->raub", weight, anchor_log[group_anchors[local]], optimize=True
        )
    return output


def run() -> None:
    folds, pos, _, energy, official_counts, actual_counts = prepare_folds()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    shifts = horizontal_shifts(pos)
    valid = energy > 0
    for fold, row in enumerate(folds):
        anchors = mapped_anchors(pos, row["val"], row["labels"], actual_fraction, official_counts)
        mask = valid.copy()
        mask[row["val"]] = False
        external_train = np.flatnonzero(mask)
        row["external_residual"] = {
            alignment: external_residual(
                pos, row["val"], row["labels"], anchors, row["query_local"],
                target, external_train, shifts, alignment,
            )
            for alignment in EXTERNAL_ALIGNMENTS
        }
        print(json.dumps({"stage": "residual", "fold": fold}), flush=True)

    gate_alpha = []
    for heldout in range(5):
        train_x = np.concatenate([folds[i]["x"] for i in range(5) if i != heldout])
        train_y = np.concatenate([folds[i]["gain_grid"] for i in range(5) if i != heldout])
        train_w = np.concatenate([folds[i]["weights"] for i in range(5) if i != heldout])
        model = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=80, max_features=0.7,
            n_jobs=-1, random_state=52180,
        )
        model.fit(train_x, train_y, sample_weight=train_w)
        gate_alpha.append(ALPHA_GRID[np.argmax(model.predict(folds[heldout]["x"]), axis=1)])

    configs = list(itertools.product(EXTERNAL_ALIGNMENTS, RESIDUAL_ALPHAS, LOCAL_SCALES, ORDERS))
    values = {config: [] for config in configs}
    baselines = []
    for fold, row in enumerate(folds):
        baseline = weighted_c1(row["base"], row["truth"], row["weights"])
        baselines.append(baseline)
        for alignment, residual_alpha, local_scale, order in configs:
            alpha = np.clip(local_scale * gate_alpha[fold], 0.0, 0.6)[:, None, None, None]
            correction = np.exp(residual_alpha * row["external_residual"][alignment])
            if order == "residual_then_local":
                corrected = normalize(row["base"] * correction)
                desired = normalize((1.0 - alpha) * corrected + alpha * row["local"])
            else:
                desired = normalize(
                    ((1.0 - alpha) * row["base"] + alpha * row["local"]) * correction
                )
            values[alignment, residual_alpha, local_scale, order].append(
                weighted_c1(desired, row["truth"], row["weights"])
            )
    baseline_array = np.asarray(baselines)
    results = []
    for config in configs:
        delta = 0.4 * (np.asarray(values[config]) - baseline_array)
        results.append(
            {
                "external_alignment": config[0], "residual_alpha": config[1],
                "local_scale": config[2], "order": config[3],
                "score_proxy_deltas": delta.tolist(),
                "mean_delta": float(delta.mean()), "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - delta.std()),
            }
        )
    results.sort(key=lambda row: (row["lcb"], row["mean_delta"]), reverse=True)
    output = {
        "official_test_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "results": results,
    }
    (ROOT / "phase9_buildable_residual_gate_pas_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "top": results[:30]}), flush=True)


if __name__ == "__main__":
    run()
