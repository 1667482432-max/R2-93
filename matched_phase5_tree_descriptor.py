from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)


def model_features(pos: np.ndarray, map_features: np.ndarray) -> np.ndarray:
    xy = pos[:, :2]
    side = xy[:, 1] > 0
    bs = np.where(side[:, None], np.array([52.0, 35.0]), np.array([-18.413, -65.881]))
    rel = xy - bs
    radius = np.linalg.norm(rel, axis=1)
    angle = np.arctan2(rel[:, 1], rel[:, 0])
    return np.column_stack(
        (
            xy,
            radius,
            np.cos(angle),
            np.sin(angle),
            side.astype(np.float32),
            map_features,
        )
    ).astype(np.float32)


@torch.no_grad()
def aggregate_descriptors(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pas_rows = []
    pdp_rows = []
    device = torch.device("cuda")
    for start in range(0, len(pred), 4):
        stop = min(start + 4, len(pred))
        x = torch.as_tensor(np.asarray(pred[start:stop]).copy(), device=device)
        pas = (torch.abs(rp.bs_fft_torch(x)) ** 2).sum((2, 3))
        pas /= torch.linalg.vector_norm(pas, dim=1, keepdim=True).clamp_min(1e-30)
        pdp = (torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2).sum((1, 2))
        pdp /= torch.linalg.vector_norm(pdp, dim=1, keepdim=True).clamp_min(1e-30)
        pas_rows.append(pas.cpu().numpy().astype(np.float32))
        pdp_rows.append(pdp.cpu().numpy().astype(np.float32))
    return np.concatenate(pas_rows), np.concatenate(pdp_rows)


def normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.maximum(rows, 0)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30
    )


def fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    rank: int,
    seed: int,
) -> np.ndarray:
    transformed = np.sqrt(np.maximum(target[train], 0))
    components = min(rank, transformed.shape[1], len(train) - 1)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    coefficient = pca.fit_transform(transformed)
    model = ExtraTreesRegressor(
        n_estimators=320,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(features[train], coefficient)
    reconstructed = pca.inverse_transform(model.predict(features[val]))
    return normalize(np.maximum(reconstructed, 0) ** 2).astype(np.float32)


def run() -> None:
    pos, _, energy = rp.load_data()
    all_pos = np.vstack((pos, np.load(ROOT / "Round2_Test_Pos.npy")))
    map_features = np.load(ROOT / "los_map_features.npy")
    features = model_features(all_pos, map_features)
    data = np.load(ROOT / "channel_descriptors.npz")
    targets = {"pas": data["pas"], "pdp": data["pdp"]}
    valid = np.flatnonzero(energy > 0)
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    records = []

    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        base = np.load(ROOT / f"matched_phase4_full_fold{fold}.npy", mmap_mode="r")
        base_descriptors = dict(zip(("pas", "pdp"), aggregate_descriptors(base)))
        predictions = {
            "pas": fit_predict(features, targets["pas"], train, val, 64, 1729 + fold),
            "pdp": fit_predict(features, targets["pdp"], train, val, 48, 2718 + fold),
        }
        for axis, prediction in predictions.items():
            np.save(ROOT / f"matched_tree_{axis}_fold{fold}.npy", prediction)
        weights = np.asarray([counts[int(group)] / np.sum(labels == group) for group in labels])
        for axis in ("pas", "pdp"):
            truth = targets[axis][val]
            for alpha in ALPHAS:
                blended = normalize(
                    (1.0 - alpha) * base_descriptors[axis] + alpha * predictions[axis]
                )
                point_score = cosine(blended, truth)
                records.append(
                    {
                        "fold": fold,
                        "axis": axis,
                        "alpha": alpha,
                        "weighted_cosine": float(np.sum(weights * point_score) / weights.sum()),
                        "mean_cosine": float(point_score.mean()),
                        "groups": {
                            int(group): float(point_score[labels == group].mean())
                            for group in np.unique(labels)
                        },
                    }
                )
        print(json.dumps({"stage": "fold", "fold": fold}), flush=True)

    summary = []
    for axis in ("pas", "pdp"):
        baseline = np.asarray(
            [
                next(
                    row["weighted_cosine"]
                    for row in records
                    if row["fold"] == fold and row["axis"] == axis and row["alpha"] == 0
                )
                for fold in range(5)
            ]
        )
        for alpha in ALPHAS:
            values = np.asarray(
                [
                    next(
                        row["weighted_cosine"]
                        for row in records
                        if row["fold"] == fold
                        and row["axis"] == axis
                        and row["alpha"] == alpha
                    )
                    for fold in range(5)
                ]
            )
            delta = values - baseline
            summary.append(
                {
                    "axis": axis,
                    "alpha": alpha,
                    "values": values.tolist(),
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    output = {"records": records, "summary": summary}
    (ROOT / "matched_phase5_tree_descriptor.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    run()
