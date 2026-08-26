from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import r2_pipeline as rp
from run_matched_metric_validation import CLEAR as BASE_CLEAR
from run_matched_metric_validation import COMMON, HybridView


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
GROUP_PROFILES = {
    3: {},
    4: {},
    5: {"R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED": "1"},
    6: {"R2_MATCHED_JOINT_METRIC_KERNEL_G6": "1"},
    8: {"R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED": "1"},
    9: {},
}
CLEAR = set(BASE_CLEAR) | {
    "R2_MATCHED_PHASE_SAFE", "R2_MULTI_START", "R2_PROJECTION_ORDER",
    "R2_FORCE_INIT", "R2_FAST_VALIDATE", "R2_SKIP_LATEST_PRED",
}


def predict(group: int, fold: int) -> None:
    output = ROOT / f"matched_phase_g{group}_fold{fold}.npy"
    labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy")
    expected = int(np.sum(labels == group))
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
    env.update({
        "R2_MATCHED_PHASE_SAFE": "1",
        "R2_ONLY_GROUP": str(group),
        "R2_VAL_FILE": f"matched_rect_val_{fold}.npy",
        "R2_VAL_GROUP_FILE": f"matched_rect_groups_{fold}.npy",
        "R2_SAVE_PRED": output.name,
        "R2_FAST_VALIDATE": "1",
        "R2_SKIP_LATEST_PRED": "1",
    })
    result = subprocess.run(
        [str(PYTHON), "r2_pipeline.py", "island-validate"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=True,
    )
    (ROOT / f"matched_phase_g{group}_fold{fold}.out").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    print(json.dumps({"group": group, "fold": fold, "cached": False}), flush=True)


def milestone_path(group: int, fold: int) -> Path:
    if group == 7:
        return ROOT / f"matched_map_g7_fold{fold}.npy"
    if group == 6:
        return ROOT / f"matched_g6_g6_fold{fold}.npy"
    return ROOT / f"matched_extended_g{group}_fold{fold}.npy"


def run() -> None:
    jobs = [(group, fold) for fold in range(5) for group in GROUP_PROFILES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job: predict(*job), jobs))

    _, channel, _ = rp.load_data()
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    test_counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    records = []
    milestone_groups = (0, 1, 5, 6, 7, 8)
    phase_groups = tuple(GROUP_PROFILES)
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy")
        weights = np.asarray([
            test_counts.get(int(group), 0) / max(1, np.sum(labels == group))
            for group in labels
        ])
        base = np.load(
            ROOT / f"matched_pred_core_nog10_safe_fold{fold}.npy", mmap_mode="r"
        )
        milestone_replacements = {
            group: np.load(milestone_path(group, fold), mmap_mode="r")
            for group in milestone_groups
        }
        milestone = rp.score_numpy_weighted(
            HybridView(base, labels, milestone_replacements), channel[val], weights
        )
        phase_replacements = dict(milestone_replacements)
        phase_replacements.update({
            group: np.load(ROOT / f"matched_phase_g{group}_fold{fold}.npy", mmap_mode="r")
            for group in phase_groups
        })
        phase = rp.score_numpy_weighted(
            HybridView(base, labels, phase_replacements), channel[val], weights
        )
        row = {
            "fold": fold, "milestone_score": milestone["score"],
            "phase_score": phase["score"],
            "delta": phase["score"] - milestone["score"],
            "milestone": milestone, "phase": phase,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
    deltas = np.asarray([row["delta"] for row in records])
    summary = {
        "milestone_scores": [row["milestone_score"] for row in records],
        "phase_scores": [row["phase_score"] for row in records],
        "deltas": deltas.tolist(),
        "mean_delta": float(deltas.mean()),
        "min_delta": float(deltas.min()),
    }
    (ROOT / "matched_phase_validation.json").write_text(
        json.dumps({"records": records, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
