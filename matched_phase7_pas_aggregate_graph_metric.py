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
GRAPH_CONFIGS = ((6, "inverse2"), (8, "inverse2"), (12, "inverse2"), (6, "exp20"))
SCALES = (0.025, 0.05, 0.10, 0.15)
MODES = (
    "xy_x025",
    "xy_x05",
    "xy_x2",
    "xy_x4",
    "bearing25",
    "bearing50",
    "bearing100",
    "polar25",
    "polar50",
    "polar100",
    "map8_w2",
    "map16_w2",
    "map16_w4",
)


def build_coordinates(pos: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    xy = pos[:, :2].astype(np.float32)
    side = xy[:, 1] > 0
    bs = np.where(
        side[:, None], np.array([52.0, 35.0]), np.array([-18.413, -65.881])
    )
    relative = xy - bs
    radius = np.linalg.norm(relative, axis=1)
    angle = np.arctan2(relative[:, 1], relative[:, 0])
    output = {}
    for name, factor in (("xy_x025", 0.25), ("xy_x05", 0.5), ("xy_x2", 2.0), ("xy_x4", 4.0)):
        output[name] = np.column_stack(
            (factor * xy[:, 0], xy[:, 1])
        ).astype(np.float32)
    for scale in (25.0, 50.0, 100.0):
        key = int(scale)
        output[f"bearing{key}"] = np.column_stack(
            (xy, scale * np.cos(angle), scale * np.sin(angle))
        ).astype(np.float32)
        output[f"polar{key}"] = np.column_stack(
            (radius, scale * np.cos(angle), scale * np.sin(angle))
        ).astype(np.float32)

    rich = np.load(ROOT / "rich_map_features.npy")[: len(pos)].astype(np.float32)
    center = np.median(rich[valid], axis=0)
    spread = np.quantile(rich[valid], 0.75, axis=0) - np.quantile(
        rich[valid], 0.25, axis=0
    )
    spread = np.maximum(spread, np.std(rich[valid], axis=0) * 0.1)
    spread = np.maximum(spread, 1e-3)
    standardized = np.clip((rich - center) / spread, -10, 10)
    latent = PCA(n_components=16, whiten=True, random_state=58900).fit_transform(
        standardized
    )
    xy_center = xy[valid].mean(0)
    xy_scale = xy[valid].std(0)
    standardized_xy = (xy - xy_center) / xy_scale
    output["map8_w2"] = np.column_stack((standardized_xy, 2.0 * latent[:, :8])).astype(np.float32)
    output["map16_w2"] = np.column_stack((standardized_xy, 2.0 * latent)).astype(np.float32)
    output["map16_w4"] = np.column_stack((standardized_xy, 4.0 * latent)).astype(np.float32)
    if set(output) != set(MODES):
        raise RuntimeError(f"coordinate mode mismatch: {sorted(output)}")
    return output


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
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        h_coefficient = fit_h_coefficients(
            unit, side, valid, train, horizontal_moment, 59000 + 10 * fold
        )
        v_coefficient = fit_v_coefficients(
            unit, side, valid, train, vertical_moment, 59100 + 10 * fold
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
                random_state=59200 + 10 * fold + VARIANTS.index(variant),
            )
            coefficient = pca.fit_transform(
                np.sqrt(np.maximum(canonical, 0))
            ).astype(np.float32)
            for mode in MODES:
                node_coordinates = coordinates[mode][nodes]
                for neighbors, kernel in GRAPH_CONFIGS:
                    predicted_coefficient = harmonic_coefficients(
                        node_coordinates,
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
                    for scale in SCALES:
                        corrected = correct(base, prediction, scale)
                        delta_point = point_cosine(corrected, truth) - baseline_point
                        records.append(
                            {
                                "fold": fold,
                                "variant": variant,
                                "mode": mode,
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
    for variant, mode, (neighbors, kernel), scale in itertools.product(
        VARIANTS, MODES, GRAPH_CONFIGS, SCALES
    ):
        selected = [
            row
            for row in records
            if row["variant"] == variant
            and row["mode"] == mode
            and row["neighbors"] == neighbors
            and row["kernel"] == kernel
            and row["scale"] == scale
        ]
        delta = np.asarray([row["delta"] for row in selected])
        common = {
            "variant": variant,
            "mode": mode,
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
    (ROOT / "matched_phase7_pas_aggregate_graph_metric.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"top": summary[:40], "safe_groups": safe_groups[:100]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
