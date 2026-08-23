from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
COMPONENTS = 192
ALPHAS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
CONFIGS = tuple(
    itertools.product(
        (8, 12, 20, 32),
        ("inverse1", "inverse2", "exp10", "exp20", "gauss10", "gauss20"),
    )
)


def edge_weight(distance: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "inverse1":
        return 1.0 / np.maximum(distance, 0.25)
    if kernel == "inverse2":
        return 1.0 / np.maximum(distance, 0.25) ** 2
    if kernel == "exp10":
        return np.exp(-distance / 10.0)
    if kernel == "exp20":
        return np.exp(-distance / 20.0)
    if kernel == "gauss10":
        return np.exp(-(distance / 10.0) ** 2)
    if kernel == "gauss20":
        return np.exp(-(distance / 20.0) ** 2)
    raise ValueError(kernel)


def harmonic_coefficients(
    xy: np.ndarray,
    train_count: int,
    known: np.ndarray,
    neighbors: int,
    kernel: str,
) -> np.ndarray:
    k = min(neighbors + 1, len(xy))
    distance, index = cKDTree(xy).query(xy, k=k)
    rows = np.repeat(np.arange(len(xy)), k - 1)
    cols = index[:, 1:].reshape(-1)
    weights = edge_weight(distance[:, 1:].reshape(-1), kernel)
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(len(xy), len(xy)))
    graph = graph.maximum(graph.T)
    degree = np.asarray(graph.sum(1)).ravel()
    unknown = np.arange(train_count, len(xy))
    known_index = np.arange(train_count)
    matrix = sparse.diags(degree[unknown]) - graph[unknown][:, unknown]
    matrix = matrix + sparse.eye(len(unknown), format="csr") * 1e-6
    rhs = graph[unknown][:, known_index] @ known
    return np.asarray(spsolve(matrix.tocsc(), rhs), dtype=np.float32)


def normalize_bs(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def point_cosine(value: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sum(value * truth, axis=1).mean((1, 2))


def run() -> None:
    pos, _, energy = rp.load_data()
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    valid = np.flatnonzero(energy > 0)
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {
        int(group): int(count)
        for group, count in zip(*np.unique(test_groups, return_counts=True))
    }
    records: dict[tuple, dict] = {}
    for config in CONFIGS:
        for alpha in ALPHAS:
            records[config, alpha] = {
                "deltas": [],
                "groups": {str(group): [] for group in counts},
            }

    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        transformed = np.sqrt(np.maximum(np.asarray(target[train]), 0)).reshape(
            len(train), -1
        )
        pca = PCA(
            n_components=COMPONENTS,
            svd_solver="randomized",
            n_oversamples=20,
            iterated_power=4,
            random_state=57100 + fold,
        )
        coefficient = pca.fit_transform(transformed).astype(np.float32)
        base = np.asarray(
            np.load(
                ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
                mmap_mode="r",
            )
        )
        truth = np.asarray(target[val])
        baseline_point = point_cosine(base, truth)
        sample_weights = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels]
        )
        nodes = np.concatenate([train, val])
        xy = pos[nodes, :2].astype(np.float32)
        for config_index, (neighbors, kernel) in enumerate(CONFIGS):
            predicted_coefficient = harmonic_coefficients(
                xy, len(train), coefficient, neighbors, kernel
            )
            reconstruction = pca.inverse_transform(predicted_coefficient)
            candidate = normalize_bs(
                np.maximum(
                    reconstruction.reshape(len(val), 256, 4, 24), 0
                )
                ** 2
            ).astype(np.float32)
            for alpha in ALPHAS:
                prediction = normalize_bs((1.0 - alpha) * base + alpha * candidate)
                delta_point = point_cosine(prediction, truth) - baseline_point
                item = records[(neighbors, kernel), alpha]
                item["deltas"].append(
                    float(
                        np.sum(sample_weights * delta_point) / sample_weights.sum()
                    )
                )
                for group in counts:
                    mask = labels == group
                    item["groups"][str(group)].append(float(delta_point[mask].mean()))
            if config_index % 6 == 5:
                print(
                    json.dumps(
                        {
                            "stage": "config",
                            "fold": fold,
                            "done": config_index + 1,
                            "total": len(CONFIGS),
                        }
                    ),
                    flush=True,
                )
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

    summary = []
    for (neighbors, kernel), alpha in records:
        raw = records[(neighbors, kernel), alpha]
        delta = np.asarray(raw["deltas"])
        summary.append(
            {
                "neighbors": neighbors,
                "kernel": kernel,
                "alpha": alpha,
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
                "groups": raw["groups"],
            }
        )
    summary.sort(key=lambda item: item["lcb"], reverse=True)
    safe_global = [item for item in summary if item["min_delta"] > 0]
    selected = {}
    for group in sorted(counts):
        options = []
        for item in summary:
            delta = np.asarray(item["groups"][str(group)])
            options.append(
                {
                    "neighbors": item["neighbors"],
                    "kernel": item["kernel"],
                    "alpha": item["alpha"],
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
        options.sort(key=lambda item: item["lcb"], reverse=True)
        safe = [item for item in options if item["min_delta"] > 0]
        selected[str(group)] = safe[0] if safe else {
            "neighbors": 0,
            "kernel": "identity",
            "alpha": 0.0,
            "deltas": [0.0] * 5,
            "mean_delta": 0.0,
            "min_delta": 0.0,
            "lcb": 0.0,
        }
    fold_delta = np.zeros(5)
    for group, count in counts.items():
        fold_delta += count * np.asarray(selected[str(group)]["deltas"])
    fold_delta /= sum(counts.values())
    combined = {
        "c1_deltas": fold_delta.tolist(),
        "score_deltas_approx": (0.4 * fold_delta).tolist(),
        "mean_score_delta_approx": float(0.4 * fold_delta.mean()),
        "min_score_delta_approx": float(0.4 * fold_delta.min()),
    }
    payload = {
        "top": summary[:100],
        "safe_global": safe_global[:100],
        "selected_by_group": selected,
        "combined": combined,
    }
    (ROOT / "matched_phase7_pas_harmonic_graph.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "top": summary[:20],
                "safe_global": safe_global[:20],
                "selected_by_group": selected,
                "combined": combined,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
