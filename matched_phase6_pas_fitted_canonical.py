from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import BANDS, build_cache
from matched_phase5_tree_descriptor import model_features


ROOT = Path(__file__).resolve().parent
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


def normalize_bs(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)


def direction(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    side = pos[:, 1] > 0
    bs = np.where(
        side[:, None],
        np.array([52.0, 35.0, 22.0]),
        np.array([-18.413, -65.881, 25.0]),
    )
    relative = pos - bs
    return relative / np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-9), side


def h_moment(target: np.ndarray, indices: np.ndarray) -> np.ndarray:
    output = []
    phase = np.exp(2j * np.pi * np.arange(16) / 16)
    for start in range(0, len(indices), 128):
        shaped = np.asarray(target[indices[start : start + 128]]).reshape(-1, 2, 16, 8, 4, BANDS)
        marginal = shaped.sum((1, 3, 4, 5))
        output.append(
            np.sum(marginal * phase[None], axis=1) / np.maximum(marginal.sum(1), 1e-30)
        )
    return np.concatenate(output)


def fit_coefficients(
    unit: np.ndarray,
    side: np.ndarray,
    valid: np.ndarray,
    train: np.ndarray,
    moment: np.ndarray,
    seed: int,
) -> dict[bool, np.ndarray]:
    lookup = np.full(len(unit), -1, dtype=np.int64)
    lookup[valid] = np.arange(len(valid))
    result = {}
    for right in (False, True):
        ids = train[side[train] == right]
        z = moment[lookup[ids]]
        weight = np.abs(z)
        phase = z / np.maximum(weight, 1e-12)
        u = unit[ids]

        def objective(coefficient: np.ndarray) -> float:
            residual = phase * np.exp(-2j * np.pi * (u @ coefficient) / 16)
            return -float(np.abs(np.sum(weight * residual) / np.sum(weight)))

        fit = differential_evolution(
            objective,
            [(-14.0, 14.0)] * 3,
            tol=1e-6,
            popsize=15,
            polish=True,
            seed=seed + int(right),
        )
        result[right] = fit.x
    return result


def shifts(unit: np.ndarray, side: np.ndarray, coefficients: dict[bool, np.ndarray]) -> np.ndarray:
    value = np.empty(len(unit), dtype=np.float64)
    for right in (False, True):
        mask = side == right
        value[mask] = unit[mask] @ coefficients[right]
    return np.rint(value).astype(np.int64)


def roll_h(rows: np.ndarray, amount: np.ndarray) -> np.ndarray:
    shaped = np.asarray(rows).reshape(len(rows), 2, 16, 8, 4, BANDS)
    output = np.empty_like(shaped)
    for index, shift in enumerate(amount):
        output[index] = np.roll(shaped[index], int(shift), axis=1)
    return output.reshape(rows.shape)


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    all_pos = np.vstack((pos, test_pos))
    basic = model_features(all_pos, np.load(ROOT / "los_map_features.npy"))
    rich = np.column_stack((basic, np.load(ROOT / "rich_map_features.npy"))).astype(np.float32)
    feature_sets = {"basic": basic, "rich": rich}
    valid = np.flatnonzero(energy > 0)
    unit, side = direction(pos)
    moment = h_moment(target, valid)
    test_groups = rp.official_island_labels(test_pos)
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        coefficient = fit_coefficients(unit, side, valid, train, moment, 40100 + 10 * fold)
        h_shift = shifts(unit, side, coefficient)
        canonical = roll_h(np.asarray(target[train]), -h_shift[train])
        transformed = np.sqrt(np.maximum(canonical, 0)).reshape(len(train), -1)
        pca = PCA(
            n_components=160,
            svd_solver="randomized",
            n_oversamples=20,
            iterated_power=4,
            random_state=40200 + fold,
        )
        pca_coefficient = pca.fit_transform(transformed)
        base = np.asarray(
            np.load(ROOT / f"matched_phase5_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
        )
        truth = np.asarray(target[val])
        weights = np.asarray([counts[int(group)] / np.sum(labels == group) for group in labels])
        baseline_point = np.sum(base * truth, axis=1).mean((1, 2))
        baseline = float(np.sum(weights * baseline_point) / weights.sum())
        for feature_name, features in feature_sets.items():
            model = ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=0.8 if feature_name == "basic" else 0.65,
                n_jobs=-1,
                random_state=40300 + 10 * fold + (feature_name == "rich"),
            )
            model.fit(features[train], pca_coefficient)
            prediction = pca.inverse_transform(model.predict(features[val]))
            prediction = np.maximum(
                roll_h(prediction.reshape(len(val), 256, 4, BANDS), h_shift[val]), 0
            ) ** 2
            prediction = normalize_bs(prediction).astype(np.float32)
            np.save(
                ROOT / f"matched_fitted_canonical_{feature_name}_pas_band{BANDS}_fold{fold}.npy",
                prediction,
            )
            for alpha in ALPHAS:
                blend = normalize_bs((1.0 - alpha) * base + alpha * prediction)
                point = np.sum(blend * truth, axis=1).mean((1, 2))
                records.append(
                    {
                        "fold": fold,
                        "feature": feature_name,
                        "alpha": alpha,
                        "delta": float(np.sum(weights * point) / weights.sum() - baseline),
                    }
                )
        print(
            json.dumps(
                {
                    "stage": "fold",
                    "fold": fold,
                    "coefficients": {str(key): value.tolist() for key, value in coefficient.items()},
                }
            ),
            flush=True,
        )

    summary = []
    for feature_name in feature_sets:
        for alpha in ALPHAS:
            delta = np.asarray(
                [row["delta"] for row in records if row["feature"] == feature_name and row["alpha"] == alpha]
            )
            summary.append(
                {
                    "feature": feature_name,
                    "alpha": alpha,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
    summary.sort(key=lambda item: item["lcb"], reverse=True)
    (ROOT / "matched_phase6_pas_fitted_canonical.json").write_text(
        json.dumps({"summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary}), flush=True)


if __name__ == "__main__":
    run()
