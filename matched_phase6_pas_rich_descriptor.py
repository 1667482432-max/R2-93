from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import BANDS, build_cache
from matched_phase5_tree_descriptor import model_features


ROOT = Path(__file__).resolve().parent
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def normalize_bs(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def run() -> None:
    pos, _, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    rich = np.load(ROOT / "rich_map_features.npy")
    features = model_features(np.vstack((pos, test_pos)), rich)
    target = build_cache()
    valid = np.flatnonzero(energy > 0)
    test_groups = rp.official_island_labels(test_pos)
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    fold_data = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        transformed = np.sqrt(np.maximum(np.asarray(target[train]), 0)).reshape(len(train), -1)
        pca = PCA(n_components=192, svd_solver="randomized", random_state=30100 + fold)
        coefficient = pca.fit_transform(transformed)
        model = ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=3,
            max_features=0.65,
            n_jobs=-1,
            random_state=30200 + fold,
        )
        model.fit(features[train], coefficient)
        prediction = pca.inverse_transform(model.predict(features[val]))
        prediction = np.maximum(
            prediction.reshape(len(val), 256, 4, BANDS), 0
        ) ** 2
        prediction = normalize_bs(prediction).astype(np.float32)
        np.save(ROOT / f"matched_rich_tree_pas_band{BANDS}_fold{fold}.npy", prediction)
        base = np.load(ROOT / f"matched_phase5_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
        truth = np.asarray(target[val])
        weights = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels], dtype=np.float64
        )
        fold_data.append((base, prediction, truth, weights))
        print(
            json.dumps(
                {
                    "stage": "fold",
                    "fold": fold,
                    "explained": float(pca.explained_variance_ratio_.sum()),
                }
            ),
            flush=True,
        )

    baseline = []
    for base, _, truth, weights in fold_data:
        cosine = np.sum(base * truth, axis=1).mean((1, 2))
        baseline.append(float(np.sum(weights * cosine) / weights.sum()))
    baseline = np.asarray(baseline)
    summary = []
    for alpha in ALPHAS:
        values = []
        for base, prediction, truth, weights in fold_data:
            blended = normalize_bs((1.0 - alpha) * np.asarray(base) + alpha * prediction)
            cosine = np.sum(blended * truth, axis=1).mean((1, 2))
            values.append(float(np.sum(weights * cosine) / weights.sum()))
        delta = np.asarray(values) - baseline
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
    (ROOT / "matched_phase6_pas_rich_descriptor.json").write_text(
        json.dumps({"summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary}), flush=True)


if __name__ == "__main__":
    run()
