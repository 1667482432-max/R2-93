from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import build_cache
from matched_phase6_pas_fitted_canonical import (
    direction,
    fit_coefficients as fit_h_coefficients,
    h_moment,
    shifts as h_shifts,
)
from matched_phase6_pas_fitted_vertical import (
    fit_coefficients as fit_v_coefficients,
    shifts as v_shifts,
    v_moment,
)
from matched_phase7_pas_aggregate_canonical import (
    aggregate,
    correct,
    normalize_profile,
    point_cosine,
    roll_profile,
)
from matched_phase7_pas_aggregate_graph_metric import build_coordinates
from matched_phase7_pas_harmonic_graph import harmonic_coefficients


ROOT = Path(__file__).resolve().parent
SCALES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
ACTIONS = {
    0: ("plain", "map8_w2", 6, "inverse2"),
    1: ("h", "polar50", 6, "exp20"),
    3: ("h", "map16_w4", 6, "exp20"),
    8: ("hv", "xy_x025", 6, "inverse2"),
    9: ("hv", "xy_x4", 8, "inverse2"),
    10: ("plain", "polar100", 6, "inverse2"),
}


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    aggregate_target = aggregate(target)
    valid = np.flatnonzero(energy > 0)
    coordinates = build_coordinates(pos, valid)
    unit, side = direction(pos)
    horizontal_moment = h_moment(target, valid)
    vertical_moment = v_moment(target, valid)
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    records = {group: {scale: [] for scale in SCALES} for group in (*ACTIONS, 6)}
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        h_coefficient = fit_h_coefficients(
            unit, side, valid, train, horizontal_moment, 59300 + 10 * fold
        )
        v_coefficient = fit_v_coefficients(
            unit, side, valid, train, vertical_moment, 59400 + 10 * fold
        )
        horizontal_shift = h_shifts(unit, side, h_coefficient)
        vertical_shift = v_shifts(unit, side, v_coefficient)
        base = np.asarray(
            np.load(
                ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
                mmap_mode="r",
            )
        )
        truth = np.asarray(target[val])
        baseline_point = point_cosine(base, truth)
        nodes = np.concatenate((train, val))
        for variant in sorted({action[0] for action in ACTIONS.values()}):
            h_train = (
                -horizontal_shift[train]
                if variant in ("h", "hv")
                else np.zeros(len(train), int)
            )
            v_train = (
                -vertical_shift[train]
                if variant == "hv"
                else np.zeros(len(train), int)
            )
            canonical = roll_profile(aggregate_target[train], h_train, v_train)
            pca = PCA(
                n_components=64,
                svd_solver="randomized",
                random_state=59500 + 10 * fold + ("plain", "h", "hv").index(variant),
            )
            coefficient = pca.fit_transform(
                np.sqrt(np.maximum(canonical, 0))
            ).astype(np.float32)
            for group, (action_variant, mode, neighbors, kernel) in ACTIONS.items():
                if action_variant != variant:
                    continue
                predicted_coefficient = harmonic_coefficients(
                    coordinates[mode][nodes],
                    len(train),
                    coefficient,
                    neighbors,
                    kernel,
                )
                prediction = pca.inverse_transform(predicted_coefficient)
                h_val = (
                    horizontal_shift[val]
                    if variant in ("h", "hv")
                    else np.zeros(len(val), int)
                )
                v_val = (
                    vertical_shift[val]
                    if variant == "hv"
                    else np.zeros(len(val), int)
                )
                prediction = normalize_profile(
                    np.maximum(roll_profile(prediction, h_val, v_val), 0) ** 2
                ).astype(np.float32)
                mask = labels == group
                for scale in SCALES:
                    corrected = correct(base, prediction, scale)
                    delta_point = point_cosine(corrected, truth) - baseline_point
                    records[group][scale].append(float(delta_point[mask].mean()))

        # The independently trained plain-RBF aggregate action is retained for g6.
        rbf = np.load(
            ROOT / f"matched_phase7_aggregate_plain_rbf_fold{fold}.npy",
            mmap_mode="r",
        )
        mask = labels == 6
        for scale in SCALES:
            corrected = correct(base, np.asarray(rbf), scale)
            delta_point = point_cosine(corrected, truth) - baseline_point
            records[6][scale].append(float(delta_point[mask].mean()))
        print(json.dumps({"stage": "fold", "fold": fold}), flush=True)

    summary = []
    selected = {}
    for group in sorted(records):
        options = []
        for scale in SCALES:
            delta = np.asarray(records[group][scale])
            options.append(
                {
                    "group": group,
                    "scale": scale,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
        options.sort(key=lambda row: row["lcb"], reverse=True)
        summary.extend(options)
        safe = [row for row in options if row["min_delta"] > 0]
        selected[str(group)] = safe[0] if safe else {
            "group": group,
            "scale": 0.0,
            "deltas": [0.0] * 5,
            "mean_delta": 0.0,
            "min_delta": 0.0,
            "lcb": 0.0,
        }
    fold_delta = np.zeros(5)
    for group, item in ((int(group), item) for group, item in selected.items()):
        fold_delta += counts[group] * np.asarray(item["deltas"])
    fold_delta /= sum(counts.values())
    combined = {
        "c1_deltas": fold_delta.tolist(),
        "score_deltas_approx": (0.4 * fold_delta).tolist(),
        "mean_score_delta_approx": float(0.4 * fold_delta.mean()),
        "min_score_delta_approx": float(0.4 * fold_delta.min()),
    }
    payload = {"actions": ACTIONS, "summary": summary, "selected": selected, "combined": combined}
    (ROOT / "matched_phase7_pas_aggregate_scale_refine.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "combined": combined}), flush=True)


if __name__ == "__main__":
    run()
