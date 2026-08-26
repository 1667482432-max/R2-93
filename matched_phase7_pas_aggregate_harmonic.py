from __future__ import annotations

import itertools
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
from matched_phase7_pas_harmonic_graph import harmonic_coefficients


ROOT = Path(__file__).resolve().parent
VARIANTS = ("plain", "h", "hv")
CONFIGS = tuple(
    itertools.product((6, 8, 12, 20), ("inverse2", "exp20", "gauss20"))
)
SCALES = (0.005, 0.01, 0.015, 0.025, 0.0375, 0.05, 0.075, 0.10, 0.15)


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    aggregate_target = aggregate(target)
    valid = np.flatnonzero(energy > 0)
    unit, side = direction(pos)
    horizontal_moment = h_moment(target, valid)
    vertical_moment = v_moment(target, valid)
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        h_coefficient = fit_h_coefficients(
            unit, side, valid, train, horizontal_moment, 58600 + 10 * fold
        )
        v_coefficient = fit_v_coefficients(
            unit, side, valid, train, vertical_moment, 58700 + 10 * fold
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
        weights = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels]
        )
        baseline_point = point_cosine(base, truth)
        nodes = np.concatenate((train, val))
        xy = pos[nodes, :2].astype(np.float32)
        for variant in VARIANTS:
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
                random_state=58800 + 10 * fold + VARIANTS.index(variant),
            )
            coefficient = pca.fit_transform(
                np.sqrt(np.maximum(canonical, 0))
            ).astype(np.float32)
            for neighbors, kernel in CONFIGS:
                predicted_coefficient = harmonic_coefficients(
                    xy, len(train), coefficient, neighbors, kernel
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
                for scale in SCALES:
                    corrected = correct(base, prediction, scale)
                    delta_point = point_cosine(corrected, truth) - baseline_point
                    records.append(
                        {
                            "fold": fold,
                            "variant": variant,
                            "neighbors": neighbors,
                            "kernel": kernel,
                            "scale": scale,
                            "delta": float(
                                np.sum(weights * delta_point) / weights.sum()
                            ),
                            "groups": {
                                str(group): float(
                                    delta_point[labels == group].mean()
                                )
                                for group in counts
                            },
                        }
                    )
        print(json.dumps({"stage": "fold", "fold": fold}), flush=True)

    summary = []
    group_summary = []
    for variant, (neighbors, kernel), scale in itertools.product(
        VARIANTS, CONFIGS, SCALES
    ):
        selected = [
            row
            for row in records
            if row["variant"] == variant
            and row["neighbors"] == neighbors
            and row["kernel"] == kernel
            and row["scale"] == scale
        ]
        delta = np.asarray([row["delta"] for row in selected])
        common = {
            "variant": variant,
            "neighbors": neighbors,
            "kernel": kernel,
            "scale": scale,
        }
        summary.append(
            {
                **common,
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
        for group in counts:
            local = np.asarray([row["groups"][str(group)] for row in selected])
            group_summary.append(
                {
                    **common,
                    "group": int(group),
                    "deltas": local.tolist(),
                    "mean_delta": float(local.mean()),
                    "min_delta": float(local.min()),
                    "lcb": float(local.mean() - 0.75 * local.std()),
                }
            )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    group_summary.sort(key=lambda row: row["lcb"], reverse=True)
    safe_groups = [row for row in group_summary if row["min_delta"] > 0]
    payload = {
        "summary": summary,
        "safe_groups": safe_groups,
        "groups": group_summary,
    }
    (ROOT / "matched_phase7_pas_aggregate_harmonic.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"top": summary[:40], "safe_groups": safe_groups[:80]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
