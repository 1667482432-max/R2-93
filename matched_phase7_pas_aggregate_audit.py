from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
SCALES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)


def normalize_bs(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def aggregate(value: np.ndarray) -> np.ndarray:
    profile = np.asarray(value).sum((2, 3))
    return profile / np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-30)


def correct(base: np.ndarray, desired: np.ndarray, scale: float) -> np.ndarray:
    current = aggregate(base)
    epsilon = 1e-3 / base.shape[1]
    ratio = np.clip((desired + epsilon) / (current + epsilon), 0.25, 4.0)
    return normalize_bs(base * ratio[:, :, None, None] ** scale)


def point_cosine(value: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sum(value * truth, axis=1).mean((1, 2))


def run() -> None:
    target_all = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        base = np.asarray(
            np.load(
                ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
                mmap_mode="r",
            )
        )
        truth = np.asarray(target_all[val])
        oracle = aggregate(truth)
        old_tree = np.asarray(
            np.load(ROOT / f"matched_tree_pas_fold{fold}.npy", mmap_mode="r")
        )
        weights = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels]
        )
        baseline = point_cosine(base, truth)
        for source, desired in (("oracle", oracle), ("old_tree", old_tree)):
            for scale in SCALES:
                prediction = correct(base, desired, scale)
                delta_point = point_cosine(prediction, truth) - baseline
                records.append(
                    {
                        "fold": fold,
                        "source": source,
                        "scale": scale,
                        "delta": float(
                            np.sum(weights * delta_point) / weights.sum()
                        ),
                        "groups": {
                            str(group): float(delta_point[labels == group].mean())
                            for group in counts
                        },
                    }
                )
        print(json.dumps({"stage": "fold", "fold": fold}), flush=True)

    summary = []
    group_summary = []
    for source in ("oracle", "old_tree"):
        for scale in SCALES:
            selected = [
                row
                for row in records
                if row["source"] == source and row["scale"] == scale
            ]
            delta = np.asarray([row["delta"] for row in selected])
            summary.append(
                {
                    "source": source,
                    "scale": scale,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
            for group in counts:
                local = np.asarray(
                    [row["groups"][str(group)] for row in selected]
                )
                group_summary.append(
                    {
                        "group": int(group),
                        "source": source,
                        "scale": scale,
                        "deltas": local.tolist(),
                        "mean_delta": float(local.mean()),
                        "min_delta": float(local.min()),
                        "lcb": float(local.mean() - 0.75 * local.std()),
                    }
                )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    group_summary.sort(key=lambda row: row["lcb"], reverse=True)
    payload = {"summary": summary, "groups": group_summary}
    (ROOT / "matched_phase7_pas_aggregate_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "top": summary,
                "old_tree_safe_groups": [
                    row
                    for row in group_summary
                    if row["source"] == "old_tree" and row["min_delta"] > 0
                ][:30],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
