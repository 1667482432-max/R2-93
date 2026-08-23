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
from matched_phase6_pas_fitted_canonical import direction, normalize_bs


ROOT = Path(__file__).resolve().parent
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15)


def v_moment(target: np.ndarray, indices: np.ndarray) -> np.ndarray:
    output = []
    phase = np.exp(2j * np.pi * np.arange(8) / 8)
    for start in range(0, len(indices), 128):
        shaped = np.asarray(target[indices[start : start + 128]]).reshape(-1, 2, 16, 8, 4, BANDS)
        marginal = shaped.sum((1, 2, 4, 5))
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
            residual = phase * np.exp(-2j * np.pi * (u @ coefficient) / 8)
            return -float(np.abs(np.sum(weight * residual) / np.sum(weight)))

        fit = differential_evolution(
            objective,
            [(-18.0, 18.0)] * 3,
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


def roll_v(rows: np.ndarray, amount: np.ndarray) -> np.ndarray:
    shaped = np.asarray(rows).reshape(len(rows), 2, 16, 8, 4, BANDS)
    output = np.empty_like(shaped)
    for index, shift in enumerate(amount):
        output[index] = np.roll(shaped[index], int(shift), axis=2)
    return output.reshape(rows.shape)


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    features = model_features(
        np.vstack((pos, test_pos)), np.load(ROOT / "los_map_features.npy")
    )
    valid = np.flatnonzero(energy > 0)
    unit, side = direction(pos)
    moment = v_moment(target, valid)
    test_groups = rp.official_island_labels(test_pos)
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        coefficient = fit_coefficients(unit, side, valid, train, moment, 42100 + 10 * fold)
        v_shift = shifts(unit, side, coefficient)
        canonical = roll_v(np.asarray(target[train]), -v_shift[train])
        transformed = np.sqrt(np.maximum(canonical, 0)).reshape(len(train), -1)
        pca = PCA(
            n_components=160,
            svd_solver="randomized",
            n_oversamples=20,
            iterated_power=4,
            random_state=42200 + fold,
        )
        pca_coefficient = pca.fit_transform(transformed)
        model = ExtraTreesRegressor(
            n_estimators=420,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=42300 + fold,
        )
        model.fit(features[train], pca_coefficient)
        prediction = pca.inverse_transform(model.predict(features[val]))
        prediction = np.maximum(
            roll_v(prediction.reshape(len(val), 256, 4, BANDS), v_shift[val]), 0
        ) ** 2
        prediction = normalize_bs(prediction).astype(np.float32)
        np.save(ROOT / f"matched_fitted_vertical_pas_band{BANDS}_fold{fold}.npy", prediction)
        base = np.asarray(
            np.load(ROOT / f"matched_phase5_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
        )
        truth = np.asarray(target[val])
        weights = np.asarray([counts[int(group)] / np.sum(labels == group) for group in labels])
        baseline = float(np.sum(weights * np.sum(base * truth, axis=1).mean((1, 2))) / weights.sum())
        for alpha in ALPHAS:
            blend = normalize_bs((1.0 - alpha) * base + alpha * prediction)
            value = float(np.sum(weights * np.sum(blend * truth, axis=1).mean((1, 2))) / weights.sum())
            records.append({"fold": fold, "alpha": alpha, "delta": value - baseline})
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
    for alpha in ALPHAS:
        delta = np.asarray([row["delta"] for row in records if row["alpha"] == alpha])
        summary.append(
            {
                "alpha": alpha,
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
    summary.sort(key=lambda item: item["lcb"], reverse=True)
    (ROOT / "matched_phase6_pas_fitted_vertical.json").write_text(
        json.dumps({"summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary}), flush=True)


if __name__ == "__main__":
    run()
