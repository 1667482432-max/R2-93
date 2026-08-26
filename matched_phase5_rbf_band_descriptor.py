from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA

import r2_pipeline as rp
from matched_phase5_tree_band_descriptor import BANDS, base_descriptor, build_cache, normalize_last
from matched_phase5_tree_descriptor import ALPHAS


ROOT = Path(__file__).resolve().parent
SAVE_BEST = os.environ.get("R2_RBF_SAVE", "0") == "1"
CONFIGS = (("linear", 100, 0.1),) if SAVE_BEST else tuple(
    (kernel, neighbors, smoothing)
    for kernel in ("thin_plate_spline", "cubic", "linear")
    for neighbors, smoothing in ((50, 0.01), (50, 0.1), (100, 0.1), (100, 1.0))
)


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    valid = np.flatnonzero(energy > 0)
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        shaped = np.asarray(target[train]).transpose(0, 2, 3, 1)
        transformed = np.sqrt(np.maximum(shaped, 0)).reshape(len(train), -1)
        pca = PCA(n_components=160, svd_solver="randomized", random_state=10100 + fold)
        coefficient = pca.fit_transform(transformed)
        scale = np.std(pos[train, :2], axis=0)
        train_xy = pos[train, :2] / scale
        val_xy = pos[val, :2] / scale
        truth = np.asarray(target[val]).transpose(0, 2, 3, 1)
        base_raw = base_descriptor(
            np.load(ROOT / f"matched_phase4_full_fold{fold}.npy", mmap_mode="r")
        )
        if SAVE_BEST:
            np.save(ROOT / f"matched_phase4_band{BANDS}_descriptor_fold{fold}.npy", base_raw)
        base = base_raw.transpose(0, 2, 3, 1)
        weights = np.asarray([counts[int(group)] / np.sum(labels == group) for group in labels])
        for kernel, neighbors, smoothing in CONFIGS:
            model = RBFInterpolator(
                train_xy,
                coefficient,
                neighbors=neighbors,
                smoothing=smoothing,
                kernel=kernel,
                degree=1 if kernel != "linear" else 0,
            )
            predicted = pca.inverse_transform(model(val_xy))
            predicted = np.maximum(predicted.reshape(len(val), 4, BANDS, 256), 0) ** 2
            predicted = normalize_last(predicted)
            if SAVE_BEST:
                np.save(
                    ROOT / f"matched_rbf_pas_band{BANDS}_fold{fold}.npy",
                    predicted.transpose(0, 3, 1, 2).astype(np.float32),
                )
            for alpha in ALPHAS:
                blend = normalize_last((1.0 - alpha) * base + alpha * predicted)
                cosine = np.sum(blend * truth, axis=3) / np.maximum(
                    np.linalg.norm(blend, axis=3) * np.linalg.norm(truth, axis=3), 1e-30
                )
                point = cosine.mean((1, 2))
                records.append(
                    {
                        "fold": fold,
                        "kernel": kernel,
                        "neighbors": neighbors,
                        "smoothing": smoothing,
                        "alpha": alpha,
                        "weighted_cosine": float(np.sum(weights * point) / weights.sum()),
                    }
                )
            print(json.dumps({"stage": "config", "fold": fold, "kernel": kernel, "neighbors": neighbors, "smoothing": smoothing}), flush=True)

    summary = []
    for kernel, neighbors, smoothing in CONFIGS:
        baseline = np.asarray(
            [next(row["weighted_cosine"] for row in records if row["fold"] == fold and row["kernel"] == kernel and row["neighbors"] == neighbors and row["smoothing"] == smoothing and row["alpha"] == 0) for fold in range(5)]
        )
        for alpha in ALPHAS:
            values = np.asarray(
                [next(row["weighted_cosine"] for row in records if row["fold"] == fold and row["kernel"] == kernel and row["neighbors"] == neighbors and row["smoothing"] == smoothing and row["alpha"] == alpha) for fold in range(5)]
            )
            delta = values - baseline
            summary.append(
                {
                    "kernel": kernel,
                    "neighbors": neighbors,
                    "smoothing": smoothing,
                    "alpha": alpha,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    (ROOT / f"matched_phase5_rbf_band{BANDS}_descriptor.json").write_text(
        json.dumps({"records": records, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:30]}), flush=True)


if __name__ == "__main__":
    run()
