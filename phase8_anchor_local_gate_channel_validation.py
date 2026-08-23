from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
from matched_phase6_physics_combo_channel_validation import update_scores
from phase8_anchor_augmented_local_pas_screen import local_prediction
from phase8_anchor_retained_pas_resolution_validation import project
from phase8_anchor_retained_pas_screen import (
    horizontal_shifts,
    mapped_anchors,
    normalize,
    weighted_c1,
)
from phase8_anchor_retained_pdp_screen import official_geometry


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
ALPHA_GRID = np.linspace(0.0, 0.6, 13, dtype=np.float32)
SCALES = (0.50, 0.75, 1.00)
ITERATIONS = (4, 12)
CONFIGS = tuple((scale, iterations) for scale in SCALES for iterations in ITERATIONS)


def sample_c1(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    return np.mean(numerator / np.maximum(denominator, 1e-30), axis=(1, 2))


def basic_gate_features(
    pos: np.ndarray,
    val: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    query_local: np.ndarray,
    train: np.ndarray,
    base: np.ndarray,
    local: np.ndarray,
) -> np.ndarray:
    query = val[query_local]
    distance, neighbor_local = cKDTree(pos[train, :2]).query(pos[query, :2], k=8)
    neighbors = train[neighbor_local]
    inner = np.isin(neighbors, val[anchors]).astype(np.float32)
    numerator = np.sum(base * local, axis=1)
    denominator = np.linalg.norm(base, axis=1) * np.linalg.norm(local, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-30)
    relative = np.zeros((len(query), 4), dtype=np.float32)
    for group in np.unique(labels[query_local]):
        rows = np.flatnonzero(labels[query_local] == group)
        group_xy = pos[val[labels == group], :2]
        lo, hi = group_xy.min(0), group_xy.max(0)
        relative[rows, :2] = (pos[query[rows], :2] - lo) / np.maximum(hi - lo, 1.0)
        relative[rows, 2:] = (
            pos[query[rows], :2] - group_xy.mean(0)
        ) / np.maximum(group_xy.std(0), 1.0)
    xy = pos[query, :2]
    side = xy[:, 1] > 0
    bs = np.where(side[:, None], np.array([52.0, 35.0]), np.array([-18.413, -65.881]))
    delta = xy - bs
    radius = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = delta / np.maximum(radius, 1e-9)
    one_hot = np.eye(11, dtype=np.float32)[labels[query_local]]
    return np.column_stack(
        [
            distance,
            inner.sum(1),
            inner[:, :4].sum(1),
            cosine.mean((1, 2)),
            cosine.std((1, 2)),
            cosine.min((1, 2)),
            np.quantile(cosine, [0.1, 0.5, 0.9], axis=(1, 2)).T,
            relative,
            radius,
            unit,
            one_hot,
        ]
    ).astype(np.float32)


def components(values: np.ndarray) -> dict[str, float]:
    c1 = values[0] / values[4]
    c2 = values[1] / values[5]
    c3 = values[2] / values[3]
    return {
        "c1_pas": float(c1),
        "c2_pdp": float(c2),
        "c3_nmse": float(c3),
        "score": float(0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3)),
    }


def prepare_folds() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    pos, channel, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    official_counts, actual_fraction, actual_counts = official_geometry(pos, energy, test_pos)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    shifts = horizontal_shifts(pos)
    valid = energy > 0
    folds = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        query_local = np.setdiff1d(np.arange(len(val)), anchors)
        query = val[query_local]
        train_mask = valid.copy()
        train_mask[val] = False
        train_mask[val[anchors]] = True
        train = np.flatnonzero(train_mask)
        base_all = np.load(ROOT / f"phase8_anchor_retained_fold{fold}_base_pas_band24.npy")
        base = base_all[query_local]
        local = local_prediction(pos[:, :2], shifts, target, train, query, 4, 3.0, "none")
        truth = np.asarray(target[query])
        score_grid = np.column_stack(
            [
                sample_c1(normalize((1.0 - alpha) * base + alpha * local), truth)
                for alpha in ALPHA_GRID
            ]
        )
        weights = np.asarray(
            [official_counts[int(group)] / np.sum(labels[query_local] == group)
             for group in labels[query_local]],
            dtype=np.float64,
        )
        folds.append(
            {
                "val": val,
                "labels": labels,
                "query_local": query_local,
                "query": query,
                "base_all": base_all,
                "base": base,
                "local": local,
                "truth": truth,
                "weights": weights,
                "x": basic_gate_features(
                    pos, val, labels, anchors, query_local, train, base, local
                ),
                "gain_grid": score_grid - score_grid[:, :1],
            }
        )
        print(json.dumps({"stage": "prepare", "fold": fold, "rows": len(query)}), flush=True)
    return folds, pos, channel, energy, official_counts, actual_counts


@torch.no_grad()
def run() -> None:
    folds, _, channel, _, official_counts, actual_counts = prepare_folds()
    alpha_by_fold = []
    for heldout in range(5):
        train_x = np.concatenate([folds[i]["x"] for i in range(5) if i != heldout])
        train_y = np.concatenate([folds[i]["gain_grid"] for i in range(5) if i != heldout])
        train_w = np.concatenate([folds[i]["weights"] for i in range(5) if i != heldout])
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=80,
            max_features=0.7,
            n_jobs=-1,
            random_state=52180,
        )
        model.fit(train_x, train_y, sample_weight=train_w)
        predicted_gain = model.predict(folds[heldout]["x"])
        alpha_by_fold.append(ALPHA_GRID[np.argmax(predicted_gain, axis=1)])

    fold_rows = []
    for fold, row in enumerate(folds):
        labels = row["labels"]
        query_local = row["query_local"]
        query = row["query"]
        weights_np = row["weights"].astype(np.float32)
        alpha = alpha_by_fold[fold]
        desired_map = {
            scale: normalize(
                (1.0 - np.clip(scale * alpha, 0.0, 0.6)[:, None, None, None]) * row["base"]
                + np.clip(scale * alpha, 0.0, 0.6)[:, None, None, None] * row["local"]
            ).astype(np.float32)
            for scale in SCALES
        }
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        accumulator = np.zeros((1 + len(CONFIGS), 6), np.float64)
        group_accumulator = np.zeros((1 + len(CONFIGS), 11, 6), np.float64)
        for start in range(0, len(query), 2):
            stop = min(start + 2, len(query))
            local_query = query_local[start:stop]
            p = torch.as_tensor(np.asarray(prediction[local_query]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(channel[query[start:stop]]).copy(), device=DEVICE)
            weights = torch.as_tensor(weights_np[start:stop], device=DEVICE)
            truth_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            truth_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
            update_scores(
                accumulator, group_accumulator, 0, p, t, truth_pas, truth_pdp,
                weights, labels[local_query]
            )
            base_band = torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE)
            for index, (scale, iterations) in enumerate(CONFIGS, 1):
                desired = torch.as_tensor(desired_map[scale][start:stop], device=DEVICE)
                value = project(p, base_band, desired, 24, iterations)
                update_scores(
                    accumulator, group_accumulator, index, value, t, truth_pas,
                    truth_pdp, weights, labels[local_query]
                )
        baseline = components(accumulator[0])
        represented = sorted(set(int(group) for group in labels[query_local]))
        rows = []
        for index, config in enumerate(CONFIGS, 1):
            value = components(accumulator[index])
            rows.append(
                {
                    "alpha_scale": config[0],
                    "iterations": config[1],
                    **value,
                    "delta": value["score"] - baseline["score"],
                    "component_deltas": {
                        key: value[key] - baseline[key]
                        for key in ("c1_pas", "c2_pdp", "c3_nmse")
                    },
                    "group_deltas": {
                        str(group): components(group_accumulator[index, group])["score"]
                        - components(group_accumulator[0, group])["score"]
                        for group in represented
                    },
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "baseline": baseline,
                "alpha_mean": float(np.average(alpha, weights=row["weights"])),
                "alpha_quantiles": np.quantile(alpha, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                "rows": rows,
            }
        )
        print(
            json.dumps({"stage": "fold", "fold": fold, "best": max(rows, key=lambda x: x["delta"])}),
            flush=True,
        )

    summary = []
    for index, config in enumerate(CONFIGS):
        deltas = [fold["rows"][index]["delta"] for fold in fold_rows]
        summary.append(
            {
                "alpha_scale": config[0],
                "iterations": config[1],
                "deltas": deltas,
                "mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)),
                "lcb": float(np.mean(deltas) - np.std(deltas)),
            }
        )
    summary.sort(key=lambda item: (item["lcb"], item["mean_delta"]), reverse=True)
    output = {
        "official_test_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "folds": fold_rows,
        "summary": summary,
    }
    (ROOT / "phase8_anchor_local_gate_channel_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    run()
