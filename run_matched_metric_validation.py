from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
MODE = os.environ.get("R2_METRIC_VALIDATION_MODE", "metric")
GROUP_PROFILES = (
    {6: {"R2_MATCHED_JOINT_METRIC_KERNEL_G6": "1"}}
    if MODE == "g6"
    else
    {7: {"R2_MATCHED_JOINT_METRIC_KERNEL_MAP": "1"}}
    if MODE == "map"
    else
    {
        group: {"R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED": "1"}
        for group in (0, 1, 5, 8)
    }
    if MODE == "extended"
    else
    {
        1: {"R2_MATCHED_JOINT_METRIC_KERNEL": "1"},
        8: {"R2_MATCHED_JOINT_METRIC_KERNEL": "1"},
    }
    if MODE == "joint"
    else {
        1: {"R2_MATCHED_FEATURE_METRIC_G1": "1"},
        8: {"R2_MATCHED_FEATURE_METRIC_G8": "1"},
    }
)
CLEAR = {
    "R2_PAS_XCORR_ALIGN", "R2_PDP_XCORR_ALIGN", "R2_ANTENNA_PROJECTION",
    "R2_DELAY_SHIFT", "R2_ANGULAR_SHIFT", "R2_DENSITY_PARAMS",
    "R2_PROJECTION_ITERS", "R2_FINAL_PAS_OVERRIDE", "R2_ONLY_GROUP",
    "R2_SHADOW_CORRECTION_SAFE", "R2_SHADOW_CORRECTION_EXTENDED",
    "R2_SHADOW_CORRECTION_CONSERVATIVE", "R2_SHADOW_CORRECTION_NOG7",
    "R2_SHADOW_ITERATION_PROFILE", "R2_SHADOW_ITERATION_PROFILE_V3",
    "R2_SHADOW_TARGET_PROFILE_V4", "R2_SHADOW_TARGET_PROFILE_V4_G7",
    "R2_SHADOW_PAS2D_PROFILE_V5", "R2_SHADOW_PAS2D_PROFILE_V5_EXTENDED",
    "R2_MATCHED_KERNEL_CORE", "R2_MATCHED_KERNEL_CORE_NOG10",
    "R2_MATCHED_TARGET_SAFE", "R2_MATCHED_FEATURE_METRIC_G1",
    "R2_MATCHED_FEATURE_METRIC_G8", "R2_MATCHED_FEATURE_METRIC_CORE",
    "R2_MATCHED_JOINT_METRIC_KERNEL",
    "R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED",
    "R2_MATCHED_JOINT_METRIC_KERNEL_MAP",
    "R2_MATCHED_JOINT_METRIC_KERNEL_G6",
    "R2_PAS_POOL", "R2_PAS_SMOOTH", "R2_PDP_POOL", "R2_PDP_SMOOTH",
    "R2_PDP_POOL_GROUPS", "R2_SAVE_PRED",
}
COMMON = {
    "R2_AFFINE_PROFILE": "1", "R2_QUADRATIC_PROFILE": "1",
    "R2_PROJECTION_ITERS_PROFILE": "1", "R2_FINAL_PAS_PROFILE": "1",
    "R2_PAS_LOW_RANK": "1", "R2_PAS_MOMENT_ALIGN": "1",
    "R2_SHADOW_CORRECTION_NOG7": "1", "R2_SHADOW_ITERATION_PROFILE_V3": "1",
    "R2_ROBUST_AMPLITUDE": "0.01", "R2_MATCHED_KERNEL_CORE_NOG10": "1",
    "R2_MATCHED_TARGET_SAFE": "1", "R2_PAS_POOL": "1", "R2_PAS_SMOOTH": "1",
    "R2_PDP_POOL": "1", "R2_PDP_SMOOTH": "1", "R2_PDP_POOL_GROUPS": "",
}


class HybridView:
    def __init__(self, base, labels, replacements):
        self.base = base
        self.labels = labels
        self.replacements = replacements
        self.lookup = {}
        for group in replacements:
            lookup = np.full(len(labels), -1, np.int64)
            rows = np.flatnonzero(labels == group)
            lookup[rows] = np.arange(len(rows))
            self.lookup[group] = lookup

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        global_rows = np.arange(len(self.base))[item]
        global_rows = np.atleast_1d(global_rows)
        output = np.asarray(self.base[item]).copy()
        if output.ndim == 3:
            output = output[None]
        for group, prediction in self.replacements.items():
            mask = self.labels[global_rows] == group
            if np.any(mask):
                output[mask] = prediction[self.lookup[group][global_rows[mask]]]
        return output


def predict(group: int, fold: int) -> None:
    output = ROOT / f"matched_{MODE}_g{group}_fold{fold}.npy"
    expected = int(np.sum(np.load(ROOT / f"matched_rect_groups_{fold}.npy") == group))
    if output.exists():
        cached = np.load(output, mmap_mode="r")
        if cached.shape == (expected, 256, 4, 192) and cached.dtype == np.complex64:
            print(json.dumps({"group": group, "fold": fold, "cached": True}), flush=True)
            return
    env = os.environ.copy()
    for name in CLEAR:
        env.pop(name, None)
    env.update(COMMON)
    env.update(GROUP_PROFILES[group])
    env["R2_ONLY_GROUP"] = str(group)
    env["R2_VAL_FILE"] = f"matched_rect_val_{fold}.npy"
    env["R2_VAL_GROUP_FILE"] = f"matched_rect_groups_{fold}.npy"
    env["R2_SAVE_PRED"] = output.name
    result = subprocess.run(
        [str(PYTHON), "r2_pipeline.py", "island-validate"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=True,
    )
    (ROOT / f"matched_{MODE}_g{group}_fold{fold}.out").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    print(json.dumps({"group": group, "fold": fold, "cached": False}), flush=True)


def prediction_path(group: int, fold: int) -> Path:
    if MODE == "g6" and group in {0, 1, 5, 7, 8}:
        return ROOT / f"matched_map_g{group}_fold{fold}.npy" if group == 7 else ROOT / f"matched_extended_g{group}_fold{fold}.npy"
    if MODE == "map" and group in {0, 1, 5, 8}:
        return ROOT / f"matched_extended_g{group}_fold{fold}.npy"
    return ROOT / f"matched_{MODE}_g{group}_fold{fold}.npy"


def run() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job: predict(*job), [
            (group, fold) for fold in range(5) for group in GROUP_PROFILES
        ]))

    _, channel, _ = rp.load_data()
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    test_counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    profiles = (
        {"g6": (6,), "g0_g1_g5_g6_g7_g8": (0, 1, 5, 6, 7, 8)}
        if MODE == "g6"
        else
        {"g7": (7,), "g0_g1_g5_g7_g8": (0, 1, 5, 7, 8)}
        if MODE == "map"
        else
        {
            "g0": (0,), "g1": (1,), "g5": (5,), "g8": (8,),
            "g0_g1_g5_g8": (0, 1, 5, 8),
        }
        if MODE == "extended"
        else {"g1": (1,), "g8": (8,), "g1_g8": (1, 8)}
    )
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy")
        weights = np.asarray([
            test_counts.get(int(group), 0) / max(1, np.sum(labels == group))
            for group in labels
        ])
        base = np.load(ROOT / f"matched_pred_core_nog10_safe_fold{fold}.npy", mmap_mode="r")
        baseline = rp.score_numpy_weighted(base, channel[val], weights)
        records.append({"fold": fold, "profile": "baseline", **baseline})
        for name, groups in profiles.items():
            replacements = {
                group: np.load(prediction_path(group, fold), mmap_mode="r")
                for group in groups
            }
            result = rp.score_numpy_weighted(
                HybridView(base, labels, replacements), channel[val], weights
            )
            records.append({"fold": fold, "profile": name, **result})
            print(json.dumps(records[-1]), flush=True)

    summary = {}
    baseline_scores = np.asarray([
        row["score"] for row in records if row["profile"] == "baseline"
    ])
    for name in profiles:
        scores = np.asarray([row["score"] for row in records if row["profile"] == name])
        delta = scores - baseline_scores
        summary[name] = {
            "scores": scores.tolist(), "deltas": delta.tolist(),
            "mean_delta": float(delta.mean()), "min_delta": float(delta.min()),
        }
    (ROOT / f"matched_{MODE}_validation.json").write_text(
        json.dumps({"records": records, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
