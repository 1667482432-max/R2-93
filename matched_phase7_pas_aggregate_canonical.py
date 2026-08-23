from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import build_cache
from matched_phase5_tree_descriptor import model_features
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


ROOT = Path(__file__).resolve().parent
VARIANTS = ("plain", "h", "hv")
MODELS = ("extra3", "extra8", "forest3", "rbf")
SCALES = (0.005, 0.01, 0.015, 0.02, 0.025, 0.0375, 0.05, 0.075, 0.10, 0.15)


def normalize_profile(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def normalize_bs(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def aggregate(value: np.ndarray) -> np.ndarray:
    return normalize_profile(np.asarray(value).sum((2, 3)))


def roll_profile(
    value: np.ndarray, h_amount: np.ndarray, v_amount: np.ndarray
) -> np.ndarray:
    shaped = np.asarray(value).reshape(len(value), 2, 16, 8)
    output = np.empty_like(shaped)
    for index, (horizontal, vertical) in enumerate(zip(h_amount, v_amount)):
        output[index] = np.roll(
            shaped[index], (int(horizontal), int(vertical)), axis=(1, 2)
        )
    return output.reshape(value.shape)


def correct(base: np.ndarray, desired: np.ndarray, scale: float) -> np.ndarray:
    current = aggregate(base)
    epsilon = 1e-3 / base.shape[1]
    ratio = np.clip((desired + epsilon) / (current + epsilon), 0.25, 4.0)
    return normalize_bs(base * ratio[:, :, None, None] ** scale)


def point_cosine(value: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sum(value * truth, axis=1).mean((1, 2))


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    aggregate_target = aggregate(target)
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    all_pos = np.vstack((pos, test_pos))
    rich_features = model_features(
        all_pos, np.load(ROOT / "rich_map_features.npy").astype(np.float32)
    )
    valid = np.flatnonzero(energy > 0)
    unit, side = direction(pos)
    horizontal_moment = h_moment(target, valid)
    vertical_moment = v_moment(target, valid)
    test_groups = rp.official_island_labels(test_pos)
    counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    records = []
    quality_records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        h_coefficient = fit_h_coefficients(
            unit, side, valid, train, horizontal_moment, 58100 + 10 * fold
        )
        v_coefficient = fit_v_coefficients(
            unit, side, valid, train, vertical_moment, 58200 + 10 * fold
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
        target_profile = aggregate_target[val]
        base_profile = aggregate(base)
        weights = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels]
        )
        baseline_point = point_cosine(base, truth)
        spatial_scale = np.std(pos[train, :2], axis=0)
        for variant in VARIANTS:
            h_train = -horizontal_shift[train] if variant in ("h", "hv") else np.zeros(len(train), int)
            v_train = -vertical_shift[train] if variant == "hv" else np.zeros(len(train), int)
            canonical = roll_profile(aggregate_target[train], h_train, v_train)
            transformed = np.sqrt(np.maximum(canonical, 0))
            pca = PCA(
                n_components=64,
                svd_solver="randomized",
                random_state=58300 + 10 * fold + VARIANTS.index(variant),
            )
            coefficient = pca.fit_transform(transformed).astype(np.float32)
            for model_name in MODELS:
                if model_name == "rbf":
                    model = RBFInterpolator(
                        pos[train, :2] / spatial_scale,
                        coefficient,
                        neighbors=100,
                        smoothing=0.1,
                        kernel="linear",
                        degree=0,
                    )
                    predicted_coefficient = model(pos[val, :2] / spatial_scale)
                else:
                    if model_name.startswith("extra"):
                        leaf = int(model_name.removeprefix("extra"))
                        model = ExtraTreesRegressor(
                            n_estimators=600,
                            min_samples_leaf=leaf,
                            max_features=0.65,
                            n_jobs=-1,
                            random_state=58400 + 100 * fold + leaf,
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=500,
                            min_samples_leaf=3,
                            max_features=0.65,
                            n_jobs=-1,
                            random_state=58500 + fold,
                        )
                    model.fit(rich_features[train], coefficient)
                    predicted_coefficient = model.predict(rich_features[val])
                prediction = pca.inverse_transform(predicted_coefficient)
                h_val = horizontal_shift[val] if variant in ("h", "hv") else np.zeros(len(val), int)
                v_val = vertical_shift[val] if variant == "hv" else np.zeros(len(val), int)
                prediction = normalize_profile(
                    np.maximum(roll_profile(prediction, h_val, v_val), 0) ** 2
                ).astype(np.float32)
                name = f"{variant}_{model_name}"
                np.save(
                    ROOT / f"matched_phase7_aggregate_{name}_fold{fold}.npy",
                    prediction,
                )
                quality_records.append(
                    {
                        "fold": fold,
                        "name": name,
                        "base_cosine": float(
                            np.mean(np.sum(base_profile * target_profile, axis=1))
                        ),
                        "candidate_cosine": float(
                            np.mean(np.sum(prediction * target_profile, axis=1))
                        ),
                    }
                )
                for scale in SCALES:
                    corrected = correct(base, prediction, scale)
                    delta_point = point_cosine(corrected, truth) - baseline_point
                    records.append(
                        {
                            "fold": fold,
                            "name": name,
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
    for name, scale in itertools.product(
        (f"{variant}_{model}" for variant in VARIANTS for model in MODELS), SCALES
    ):
        selected = [
            row for row in records if row["name"] == name and row["scale"] == scale
        ]
        delta = np.asarray([row["delta"] for row in selected])
        summary.append(
            {
                "name": name,
                "scale": scale,
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
                    "group": int(group),
                    "name": name,
                    "scale": scale,
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
        "quality": quality_records,
        "summary": summary,
        "safe_groups": safe_groups,
        "groups": group_summary,
    }
    (ROOT / "matched_phase7_pas_aggregate_canonical.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"top": summary[:40], "safe_groups": safe_groups[:60]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
