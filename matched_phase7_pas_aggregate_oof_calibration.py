from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import build_cache
from matched_phase7_pas_aggregate_canonical import (
    aggregate,
    correct,
    normalize_profile,
    point_cosine,
)


ROOT = Path(__file__).resolve().parent
METHODS = (
    "sqrt_fold_mean",
    "sqrt_fold_median",
    "log_fold_mean",
    "log_fold_median",
    "target_fold_mean",
)
SCALES = (0.005, 0.01, 0.015, 0.025, 0.0375, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def fold_statistic(rows: list[np.ndarray], median: bool) -> np.ndarray:
    values = np.stack(rows)
    return np.median(values, axis=0) if median else values.mean(axis=0)


def desired_profiles(
    method: str,
    held: dict,
    training: list[dict],
    groups: list[int],
) -> np.ndarray:
    base = held["base_profile"]
    base_sqrt = np.sqrt(np.maximum(base, 0))
    output = base.copy()
    epsilon = 1e-4 / base.shape[1]
    for group in groups:
        mask = held["labels"] == group
        statistics = []
        for row in training:
            source = row["labels"] == group
            if method.startswith("sqrt_"):
                value = np.sqrt(np.maximum(row["truth_profile"][source], 0)) - np.sqrt(
                    np.maximum(row["base_profile"][source], 0)
                )
            elif method.startswith("log_"):
                value = np.log(
                    (row["truth_profile"][source] + epsilon)
                    / (row["base_profile"][source] + epsilon)
                )
                value = np.clip(value, -3.0, 3.0)
            elif method == "target_fold_mean":
                value = np.sqrt(np.maximum(row["truth_profile"][source], 0))
            else:
                raise ValueError(method)
            statistics.append(value.mean(axis=0))
        statistic = fold_statistic(statistics, method.endswith("median"))
        if method.startswith("sqrt_"):
            candidate = np.maximum(base_sqrt[mask] + statistic, 0) ** 2
        elif method.startswith("log_"):
            candidate = base[mask] * np.exp(statistic)
        else:
            candidate = np.broadcast_to(statistic, (int(mask.sum()), len(statistic))) ** 2
        output[mask] = normalize_profile(candidate)
    return output.astype(np.float32)


def run() -> None:
    target = build_cache()
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {
        int(group): int(count)
        for group, count in zip(*np.unique(test_groups, return_counts=True))
    }
    groups = sorted(counts)
    folds = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        base = np.asarray(
            np.load(
                ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
                mmap_mode="r",
            )
        )
        truth = np.asarray(target[val])
        folds.append(
            {
                "fold": fold,
                "labels": labels,
                "base": base,
                "truth": truth,
                "base_profile": aggregate(base),
                "truth_profile": aggregate(truth),
                "baseline": point_cosine(base, truth),
                "weights": np.asarray(
                    [counts[int(group)] / np.sum(labels == group) for group in labels]
                ),
            }
        )

    candidates: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    for holdout, held in enumerate(folds):
        training = [row for index, row in enumerate(folds) if index != holdout]
        for method in METHODS:
            candidates[method].append(
                desired_profiles(method, held, training, groups)
            )
        print(json.dumps({"stage": "fold_candidate", "fold": holdout}), flush=True)

    summary = []
    group_summary = []
    for method, scale in itertools.product(METHODS, SCALES):
        deltas = []
        local = {group: [] for group in groups}
        for held, desired in zip(folds, candidates[method]):
            prediction = correct(held["base"], desired, scale)
            delta_point = point_cosine(prediction, held["truth"]) - held["baseline"]
            deltas.append(
                float(np.sum(held["weights"] * delta_point) / held["weights"].sum())
            )
            for group in groups:
                local[group].append(float(delta_point[held["labels"] == group].mean()))
        delta = np.asarray(deltas)
        common = {"method": method, "scale": scale}
        summary.append(
            {
                **common,
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
        for group in groups:
            value = np.asarray(local[group])
            group_summary.append(
                {
                    **common,
                    "group": group,
                    "deltas": value.tolist(),
                    "mean_delta": float(value.mean()),
                    "min_delta": float(value.min()),
                    "lcb": float(value.mean() - 0.75 * value.std()),
                }
            )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    group_summary.sort(key=lambda row: row["lcb"], reverse=True)
    safe_groups = [row for row in group_summary if row["min_delta"] > 0]
    payload = {
        "summary": summary,
        "safe_groups": safe_groups,
        "groups": group_summary,
    }
    (ROOT / "matched_phase7_pas_aggregate_oof_calibration.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(
        json.dumps(
            {"top": summary[:20], "safe_groups": safe_groups[:50]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
