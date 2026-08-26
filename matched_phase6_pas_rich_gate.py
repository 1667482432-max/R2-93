from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import r2_pipeline as rp
from matched_spatial_gate import geometry_features

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import BANDS, build_cache


ROOT = Path(__file__).resolve().parent
ALPHA_GRID = np.linspace(0.0, 1.0, 41)


def descriptor_stats(value: np.ndarray) -> np.ndarray:
    shaped = np.asarray(value).reshape(len(value), 2, 16, 8, 4, BANDS)
    probability = shaped / np.maximum(shaped.sum((1, 2, 3), keepdims=True), 1e-30)
    flat = probability.reshape(len(value), 256, 4, BANDS)
    entropy = -np.sum(flat * np.log(np.maximum(flat, 1e-30)), axis=1) / np.log(256)
    peak = flat.max(1)
    h = probability.sum((1, 3))
    v = probability.sum((1, 2))
    h_phase = np.exp(2j * np.pi * np.arange(16) / 16).astype(np.complex64)
    v_phase = np.exp(2j * np.pi * np.arange(8) / 8).astype(np.complex64)
    h_moment = np.sum(h * h_phase[None, :, None, None], axis=1)
    v_moment = np.sum(v * v_phase[None, :, None, None], axis=1)
    blocks = []
    for item in (entropy, peak, np.abs(h_moment), np.abs(v_moment)):
        blocks.append(
            np.column_stack(
                (
                    item.mean((1, 2)),
                    item.std((1, 2)),
                    np.quantile(item, 0.1, axis=(1, 2)),
                    np.quantile(item, 0.5, axis=(1, 2)),
                    np.quantile(item, 0.9, axis=(1, 2)),
                )
            )
        )
    blocks.append(
        np.column_stack(
            (
                np.real(h_moment).mean((1, 2)),
                np.imag(h_moment).mean((1, 2)),
                np.real(v_moment).mean((1, 2)),
                np.imag(v_moment).mean((1, 2)),
            )
        )
    )
    return np.column_stack(blocks).astype(np.float32)


def similarity_stats(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    value = np.sum(left * right, axis=1)
    return np.column_stack(
        (
            value.mean((1, 2)),
            value.std((1, 2)),
            value.min((1, 2)),
            np.quantile(value, 0.1, axis=(1, 2)),
            np.quantile(value, 0.5, axis=(1, 2)),
            np.quantile(value, 0.9, axis=(1, 2)),
            value.max((1, 2)),
        )
    ).astype(np.float32)


def component_scores(
    base: np.ndarray, candidate: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bt = np.sum(base * truth, axis=1)
    ct = np.sum(candidate * truth, axis=1)
    bc = np.sum(base * candidate, axis=1)
    scores = []
    for alpha in ALPHA_GRID:
        denominator = np.sqrt(
            np.maximum(
                (1.0 - alpha) ** 2
                + alpha**2
                + 2.0 * alpha * (1.0 - alpha) * bc,
                1e-30,
            )
        )
        scores.append((((1.0 - alpha) * bt + alpha * ct) / denominator).mean((1, 2)))
    scores = np.stack(scores, axis=1)
    baseline = scores[:, 0]
    best_index = scores.argmax(1)
    optimal_alpha = ALPHA_GRID[best_index]
    oracle_gain = scores[np.arange(len(scores)), best_index] - baseline
    return bt, ct, bc, optimal_alpha, oracle_gain


def evaluate_alpha(row: dict, alpha: np.ndarray) -> float:
    alpha = np.asarray(alpha)[:, None, None]
    denominator = np.sqrt(
        np.maximum(
            (1.0 - alpha) ** 2
            + alpha**2
            + 2.0 * alpha * (1.0 - alpha) * row["bc"],
            1e-30,
        )
    )
    cosine = ((1.0 - alpha) * row["bt"] + alpha * row["ct"]) / denominator
    point = cosine.mean((1, 2))
    return float(np.sum(row["weights"] * point) / row["weights"].sum())


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    rich_map = np.load(ROOT / "rich_map_features.npy")
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    folds = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        paths = [
            ROOT / f"matched_phase5_pas_band{BANDS}_fold{fold}.npy",
            ROOT / f"matched_rich_tree_pas_band{BANDS}_fold{fold}.npy",
            ROOT / f"matched_tree_pas_band{BANDS}_fold{fold}.npy",
            ROOT / f"matched_rbf_pas_band{BANDS}_fold{fold}.npy",
            ROOT / f"matched_mlp_pas_band{BANDS}_fold{fold}.npy",
            ROOT / f"matched_tree_pas_canonical_c1_k5p0_fold{fold}.npy",
        ]
        components = [np.asarray(np.load(path, mmap_mode="r")) for path in paths]
        base, candidate = components[:2]
        truth = np.asarray(target[val])
        bt, ct, bc, optimal_alpha, oracle_gain = component_scores(base, candidate, truth)
        geometry = geometry_features(pos, energy, fold, val, labels)
        feature_blocks = [geometry, rich_map[val]]
        feature_blocks.extend(descriptor_stats(component) for component in components)
        for left, right in itertools.combinations(range(len(components)), 2):
            feature_blocks.append(similarity_stats(components[left], components[right]))
        x = np.column_stack(feature_blocks).astype(np.float32)
        weights = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels], dtype=np.float64
        )
        folds.append(
            {
                "fold": fold,
                "x": x,
                "labels": labels,
                "optimal_alpha": optimal_alpha,
                "oracle_gain": oracle_gain,
                "bt": bt,
                "ct": ct,
                "bc": bc,
                "weights": weights,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "fold",
                    "fold": fold,
                    "features": int(x.shape[1]),
                    "oracle": float(np.sum(weights * oracle_gain) / weights.sum()),
                    "positive": float(np.mean(oracle_gain > 0)),
                }
            ),
            flush=True,
        )

    np.savez_compressed(
        ROOT / "matched_phase6_pas_rich_gate_features.npz",
        **{f"x{row['fold']}": row["x"] for row in folds},
        **{f"labels{row['fold']}": row["labels"] for row in folds},
        **{f"optimal_alpha{row['fold']}": row["optimal_alpha"] for row in folds},
        **{f"oracle_gain{row['fold']}": row["oracle_gain"] for row in folds},
        **{f"bt{row['fold']}": row["bt"] for row in folds},
        **{f"ct{row['fold']}": row["ct"] for row in folds},
        **{f"bc{row['fold']}": row["bc"] for row in folds},
        **{f"weights{row['fold']}": row["weights"] for row in folds},
    )
    if os.environ.get("R2_GATE_PREPARE_ONLY") == "1":
        print(json.dumps({"stage": "cache", "prepared": True}), flush=True)
        return

    baselines = np.asarray([evaluate_alpha(row, np.zeros(len(row["x"]))) for row in folds])
    predictions: dict[str, list[np.ndarray]] = {}
    for leaf in (5, 10, 20, 40, 80):
        key = f"reg_leaf{leaf}"
        predictions[key] = []
        for holdout, row in enumerate(folds):
            train = [other for index, other in enumerate(folds) if index != holdout]
            model = ExtraTreesRegressor(
                n_estimators=700,
                min_samples_leaf=leaf,
                max_features=0.65,
                n_jobs=-1,
                random_state=31100 + leaf + holdout,
            )
            model.fit(
                np.concatenate([item["x"] for item in train]),
                np.concatenate([item["optimal_alpha"] for item in train]),
                sample_weight=np.concatenate([item["weights"] for item in train]),
            )
            predictions[key].append(np.clip(model.predict(row["x"]), 0.0, 1.0))

    for threshold in (0.001, 0.003, 0.005, 0.01):
        for leaf in (5, 10, 20, 40, 80):
            key = f"cls_t{threshold}_leaf{leaf}"
            predictions[key] = []
            for holdout, row in enumerate(folds):
                train = [other for index, other in enumerate(folds) if index != holdout]
                model = ExtraTreesClassifier(
                    n_estimators=700,
                    min_samples_leaf=leaf,
                    max_features=0.65,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=31200 + leaf + holdout,
                )
                model.fit(
                    np.concatenate([item["x"] for item in train]),
                    np.concatenate([item["oracle_gain"] > threshold for item in train]),
                    sample_weight=np.concatenate([item["weights"] for item in train]),
                )
                predictions[key].append(model.predict_proba(row["x"])[:, 1])

    summary = []
    for name, prediction in predictions.items():
        if name.startswith("reg"):
            mappings = [
                (f"shrink_{scale}", lambda p, scale=scale: np.clip(scale * p, 0, 1))
                for scale in (0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
            ]
        else:
            mappings = []
            for scale in (0.10, 0.20, 0.30, 0.50, 0.75, 1.0):
                mappings.append((f"soft_{scale}", lambda p, scale=scale: scale * p))
                for probability in (0.50, 0.60, 0.70):
                    mappings.append(
                        (
                            f"hard_{scale}_{probability}",
                            lambda p, scale=scale, probability=probability: scale
                            * (p >= probability),
                        )
                    )
        for mapping_name, mapping in mappings:
            values = np.asarray(
                [evaluate_alpha(row, mapping(p)) for row, p in zip(folds, prediction)]
            )
            delta = values - baselines
            summary.append(
                {
                    "model": name,
                    "mapping": mapping_name,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                    "mean_alpha": [float(np.mean(mapping(p))) for p in prediction],
                }
            )
    summary.sort(key=lambda item: item["lcb"], reverse=True)
    (ROOT / "matched_phase6_pas_rich_gate.json").write_text(
        json.dumps({"summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:50]}), flush=True)


if __name__ == "__main__":
    run()
