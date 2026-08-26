from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
from matched_phase5_tree_descriptor import ALPHAS, model_features


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "pdp_ue_descriptors.npy"


@torch.no_grad()
def build_cache() -> np.ndarray:
    if CACHE.exists():
        return np.load(CACHE, mmap_mode="r")
    _, channel, energy = rp.load_data()
    building = CACHE.with_suffix(".building.npy")
    output = np.lib.format.open_memmap(
        building, mode="w+", dtype=np.float32, shape=(len(channel), 4, 192)
    )
    device = torch.device("cuda")
    for start in range(0, len(channel), 8):
        stop = min(start + 8, len(channel))
        x = torch.as_tensor(np.asarray(channel[start:stop]).copy(), device=device)
        pdp = torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2
        pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
        pdp = pdp.mean(1)
        pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
        output[start:stop] = pdp.cpu().numpy().astype(np.float32)
        if stop % 400 == 0 or stop == len(channel):
            output.flush()
            print(json.dumps({"stage": "cache", "done": stop}), flush=True)
    del output
    check = np.load(building, mmap_mode="r")
    zero_rows = np.sum(np.abs(check), axis=(1, 2)) == 0
    if check.shape != (len(channel), 4, 192):
        raise RuntimeError(f"Unexpected cache shape: {check.shape}")
    if not np.array_equal(zero_rows, energy == 0):
        raise RuntimeError("PDP UE cache zero rows do not match zero-channel outliers")
    if not np.isfinite(check).all():
        raise RuntimeError("PDP UE cache contains non-finite values")
    del check
    building.replace(CACHE)
    return np.load(CACHE, mmap_mode="r")


@torch.no_grad()
def base_descriptor(pred: np.ndarray) -> np.ndarray:
    rows = []
    device = torch.device("cuda")
    for start in range(0, len(pred), 4):
        stop = min(start + 4, len(pred))
        x = torch.as_tensor(np.asarray(pred[start:stop]).copy(), device=device)
        pdp = torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2
        pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
        pdp = pdp.mean(1)
        pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
        rows.append(pdp.cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def normalize_last(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-30)


def fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
) -> np.ndarray:
    transformed = np.sqrt(np.maximum(np.asarray(target[train]), 0)).reshape(len(train), -1)
    pca = PCA(n_components=96, svd_solver="randomized", random_state=seed)
    coefficient = pca.fit_transform(transformed)
    model = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(features[train], coefficient)
    predicted = pca.inverse_transform(model.predict(features[val]))
    predicted = np.maximum(predicted.reshape(len(val), 4, 192), 0) ** 2
    return normalize_last(predicted).astype(np.float32)


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    features = model_features(
        np.vstack((pos, test_pos)), np.load(ROOT / "los_map_features.npy")
    )
    valid = np.flatnonzero(energy > 0)
    test_groups = rp.official_island_labels(test_pos)
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        base = base_descriptor(
            np.load(ROOT / f"matched_phase4_full_fold{fold}.npy", mmap_mode="r")
        )
        prediction_path = ROOT / f"matched_tree_pdp_ue_fold{fold}.npy"
        if prediction_path.exists():
            prediction = np.load(prediction_path)
        else:
            prediction = fit_predict(features, target, train, val, 6151 + fold)
            np.save(prediction_path, prediction)
        weights = np.asarray([counts[int(group)] / np.sum(labels == group) for group in labels])
        truth = np.asarray(target[val])
        for alpha in ALPHAS:
            blend = normalize_last((1.0 - alpha) * base + alpha * prediction)
            cosine = np.sum(blend * truth, axis=2) / np.maximum(
                np.linalg.norm(blend, axis=2) * np.linalg.norm(truth, axis=2), 1e-30
            )
            point = cosine.mean(1)
            records.append(
                {
                    "fold": fold,
                    "alpha": alpha,
                    "weighted_cosine": float(np.sum(weights * point) / weights.sum()),
                    "mean_cosine": float(point.mean()),
                    "groups": {
                        int(group): float(point[labels == group].mean())
                        for group in np.unique(labels)
                    },
                }
            )
        print(json.dumps({"stage": "fold", "fold": fold}), flush=True)

    baseline = np.asarray(
        [next(row["weighted_cosine"] for row in records if row["fold"] == fold and row["alpha"] == 0) for fold in range(5)]
    )
    summary = []
    for alpha in ALPHAS:
        values = np.asarray(
            [next(row["weighted_cosine"] for row in records if row["fold"] == fold and row["alpha"] == alpha) for fold in range(5)]
        )
        delta = values - baseline
        summary.append(
            {
                "alpha": alpha,
                "values": values.tolist(),
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
    group_summary = []
    for group in sorted(counts):
        group_baseline = np.asarray(
            [
                next(
                    row["groups"][group]
                    for row in records
                    if row["fold"] == fold and row["alpha"] == 0
                )
                for fold in range(5)
            ]
        )
        for alpha in ALPHAS:
            values = np.asarray(
                [
                    next(
                        row["groups"][group]
                        for row in records
                        if row["fold"] == fold and row["alpha"] == alpha
                    )
                    for fold in range(5)
                ]
            )
            delta = values - group_baseline
            group_summary.append(
                {
                    "group": group,
                    "alpha": alpha,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    group_summary.sort(key=lambda row: row["lcb"], reverse=True)
    (ROOT / "matched_phase5_tree_pdp_ue_descriptor.json").write_text(
        json.dumps(
            {"records": records, "summary": summary, "group_summary": group_summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "groups": group_summary[:30]}), flush=True)


if __name__ == "__main__":
    run()
