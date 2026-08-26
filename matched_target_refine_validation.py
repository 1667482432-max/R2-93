from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import r2_pipeline as rp
from run_matched_metric_validation import CLEAR as BASE_CLEAR, COMMON


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
CONFIGS = (
    ("g1_pas_pool_both025", 1, {"R2_PAS_POOL_OVERRIDE": "both:0.25"}),
    ("g1_pas_smooth_hv010", 1, {"R2_PAS_SMOOTH_OVERRIDE": "hv3:0.10"}),
    ("g1_pas_smooth_hv025", 1, {"R2_PAS_SMOOTH_OVERRIDE": "hv3:0.25"}),
    ("g5_pas_smooth_hv010", 5, {"R2_PAS_SMOOTH_OVERRIDE": "hv3:0.10"}),
    ("g7_pdp_smooth_d5020", 7, {"R2_PDP_SMOOTH_OVERRIDE": "d5:0.20"}),
    ("g1_pas_pool_both050", 1, {"R2_PAS_POOL_OVERRIDE": "both:0.50"}),
    ("g1_pas_smooth_hv050", 1, {"R2_PAS_SMOOTH_OVERRIDE": "hv3:0.50"}),
    ("g1_pas_pool025_smooth025", 1, {
        "R2_PAS_POOL_OVERRIDE": "both:0.25",
        "R2_PAS_SMOOTH_OVERRIDE": "hv3:0.25",
    }),
    ("g1_pdp_pool_both070", 1, {"R2_PDP_POOL_OVERRIDE": "both:0.70"}),
    ("g1_target_combo", 1, {
        "R2_PAS_POOL_OVERRIDE": "both:0.25",
        "R2_PAS_SMOOTH_OVERRIDE": "hv3:0.25",
        "R2_PDP_POOL_OVERRIDE": "both:0.70",
    }),
    ("g7_pdp_smooth_d5010", 7, {"R2_PDP_SMOOTH_OVERRIDE": "d5:0.10"}),
    ("g9_pas_pool_sc100", 9, {"R2_PAS_POOL_OVERRIDE": "sc:1.0"}),
    ("g7_pas_pool_add025", 7, {"R2_PAS_POOL_OVERRIDE": "add:0.25"}),
    ("g7_pas_pool_add050", 7, {"R2_PAS_POOL_OVERRIDE": "add:0.50"}),
    ("g7_pas_pool_add075", 7, {"R2_PAS_POOL_OVERRIDE": "add:0.75"}),
    ("g1_add_target_combo", 1, {
        "R2_PAS_POOL_OVERRIDE": "add:0.25",
        "R2_PAS_SMOOTH_OVERRIDE": "hv3:0.25",
        "R2_PDP_POOL_OVERRIDE": "both:0.70",
    }),
    ("g5_pas_pool_add050", 5, {"R2_PAS_POOL_OVERRIDE": "add:0.50"}),
    ("g7_pdp_pool_add025", 7, {"R2_PDP_POOL_OVERRIDE": "add:0.25"}),
    ("g7_pdp_pool_add050", 7, {"R2_PDP_POOL_OVERRIDE": "add:0.50"}),
    ("g4_pdp_pool_add025", 4, {"R2_PDP_POOL_OVERRIDE": "add:0.25"}),
    ("g3_pdp_pool_add025", 3, {"R2_PDP_POOL_OVERRIDE": "add:0.25"}),
    ("g7_pas_pool_bothmed025", 7, {"R2_PAS_POOL_OVERRIDE": "both_med:0.25"}),
    ("g7_pas_pool_bothmed050", 7, {"R2_PAS_POOL_OVERRIDE": "both_med:0.50"}),
    ("g5_pas_pool_scmed100", 5, {"R2_PAS_POOL_OVERRIDE": "sc_med:1.0"}),
    ("g1_med_target_combo", 1, {
        "R2_PAS_POOL_OVERRIDE": "both_med:0.25",
        "R2_PAS_SMOOTH_OVERRIDE": "hv3:0.25",
        "R2_PDP_POOL_OVERRIDE": "both:0.70",
    }),
    ("g4_pdp_pool_hvmed085", 4, {"R2_PDP_POOL_OVERRIDE": "hv_med:0.85"}),
    ("g7_pdp_pool_hvmed085", 7, {"R2_PDP_POOL_OVERRIDE": "hv_med:0.85"}),
    ("g3_pdp_pool_hvmed100", 3, {"R2_PDP_POOL_OVERRIDE": "hv_med:1.0"}),
    ("g9_pdp_pool_hvmed100", 9, {"R2_PDP_POOL_OVERRIDE": "hv_med:1.0"}),
    ("g10_pdp_pool_bothmed085", 10, {"R2_PDP_POOL_OVERRIDE": "both_med:0.85"}),
    ("g3_affine_pas075_001", 3, {"R2_AFFINE_PAS_OVERRIDE": "0.75:0.01"}),
    ("g5_affine_pas075_010_smooth", 5, {
        "R2_AFFINE_PAS_OVERRIDE": "0.75:0.10",
        "R2_PAS_SMOOTH_OVERRIDE": "hv3:0.10",
    }),
    ("g10_affine_pas075_010", 10, {"R2_AFFINE_PAS_OVERRIDE": "0.75:0.10"}),
    ("g0_affine_pas075_1000", 0, {"R2_AFFINE_PAS_OVERRIDE": "0.75:10.0"}),
    ("g4_affine_pdp000_100", 4, {"R2_AFFINE_PDP_OVERRIDE": "0.0:1.0"}),
    ("g9_affine_pas075_001", 9, {"R2_AFFINE_PAS_OVERRIDE": "0.75:0.01"}),
    ("g1_affine_pas025_010_combo", 1, {
        "R2_AFFINE_PAS_OVERRIDE": "0.25:0.10",
        "R2_PAS_POOL_OVERRIDE": "both:0.25",
        "R2_PAS_SMOOTH_OVERRIDE": "hv3:0.25",
        "R2_PDP_POOL_OVERRIDE": "both:0.70",
    }),
    ("g0_refine10_lr0003", 0, {
        "R2_REFINE_STEPS": "10", "R2_REFINE_LR": "0.003",
    }),
    ("g3_refine10_lr0003", 3, {
        "R2_REFINE_STEPS": "10", "R2_REFINE_LR": "0.003",
    }),
    ("g4_refine10_lr0003", 4, {
        "R2_REFINE_STEPS": "10", "R2_REFINE_LR": "0.003",
    }),
    ("g7_refine10_lr0003", 7, {
        "R2_REFINE_STEPS": "10", "R2_REFINE_LR": "0.003",
    }),
    ("g9_refine10_lr0003", 9, {
        "R2_REFINE_STEPS": "10", "R2_REFINE_LR": "0.003",
    }),
    ("g10_refine10_lr0003", 10, {
        "R2_REFINE_STEPS": "10", "R2_REFINE_LR": "0.003",
    }),
    ("g10_kernel_pas8_200_080", 10, {
        "R2_KERNEL_PAS_OVERRIDE": "8:2.0:0.8",
    }),
    ("g10_kernel_pdp24_125_050", 10, {
        "R2_KERNEL_PDP_OVERRIDE": "24:1.25:0.5",
    }),
    ("g10_kernel_both_matched", 10, {
        "R2_KERNEL_PAS_OVERRIDE": "8:2.0:0.8",
        "R2_KERNEL_PDP_OVERRIDE": "24:1.25:0.5",
    }),
)
CLEAR = set(BASE_CLEAR) | {
    "R2_MATCHED_PHASE_SAFE",
    "R2_PAS_POOL_OVERRIDE",
    "R2_PAS_SMOOTH_OVERRIDE",
    "R2_PDP_POOL_OVERRIDE",
    "R2_PDP_SMOOTH_OVERRIDE",
    "R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED",
    "R2_MATCHED_JOINT_METRIC_KERNEL_MAP",
    "R2_MATCHED_JOINT_METRIC_KERNEL_G6",
    "R2_AFFINE_PAS_OVERRIDE",
    "R2_AFFINE_PDP_OVERRIDE",
    "R2_REFINE_STEPS",
    "R2_REFINE_LR",
    "R2_KERNEL_PAS_OVERRIDE",
    "R2_KERNEL_PDP_OVERRIDE",
}


def metric_profile(group: int) -> dict[str, str]:
    if group in (1, 5):
        return {"R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED": "1"}
    if group == 7:
        return {"R2_MATCHED_JOINT_METRIC_KERNEL_MAP": "1"}
    return {"R2_MATCHED_JOINT_METRIC_KERNEL_G6": "1"}


def evaluate(name: str, group: int, overrides: dict[str, str], fold: int) -> dict:
    labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy")
    count = int(np.sum(labels == group))
    output = ROOT / f"matched_refine_{name}_fold{fold}.npy"
    if not output.exists() or np.load(output, mmap_mode="r").shape != (count, 256, 4, 192):
        env = os.environ.copy()
        for key in CLEAR:
            env.pop(key, None)
        env.update(COMMON)
        env.update(metric_profile(group))
        env.update(overrides)
        env.update(
            {
                "R2_MATCHED_PHASE_SAFE": "1",
                "R2_ONLY_GROUP": str(group),
                "R2_VAL_FILE": f"matched_rect_val_{fold}.npy",
                "R2_VAL_GROUP_FILE": f"matched_rect_groups_{fold}.npy",
                "R2_SAVE_PRED": output.name,
                "R2_FAST_VALIDATE": "1",
                "R2_SKIP_LATEST_PRED": "1",
            }
        )
        subprocess.run(
            [str(PYTHON), "r2_pipeline.py", "island-validate"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    _, channel, _ = rp.load_data()
    val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
    target = channel[val[labels == group]]
    score = rp.score_numpy(np.load(output, mmap_mode="r"), target)
    result = {"name": name, "group": group, "fold": fold, **score}
    print(json.dumps(result), flush=True)
    return result


def run() -> None:
    requested = {value for value in os.environ.get("R2_REFINE_NAMES", "").split(",") if value}
    configs = [config for config in CONFIGS if not requested or config[0] in requested]
    jobs = [(*config, fold) for config in configs for fold in range(5)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda job: evaluate(*job), jobs))
    phase2 = [
        np.load(ROOT / f"matched_phase2_full_fold{fold}.npy", mmap_mode="r")
        for fold in range(5)
    ]
    _, channel, _ = rp.load_data()
    summary = []
    for name, group, _ in configs:
        baseline = []
        candidate = []
        for fold in range(5):
            val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
            labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy")
            mask = labels == group
            baseline.append(rp.score_numpy(phase2[fold][mask], channel[val[mask]])["score"])
            candidate.append(
                next(
                    row["score"]
                    for row in records
                    if row["name"] == name and row["fold"] == fold
                )
            )
        delta = np.asarray(candidate) - baseline
        summary.append(
            {
                "name": name,
                "group": group,
                "baseline": baseline,
                "scores": candidate,
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "positive": int(np.sum(delta > 0)),
            }
        )
    summary.sort(key=lambda row: row["mean_delta"], reverse=True)
    (ROOT / "matched_target_refine_validation.json").write_text(
        json.dumps({"records": records, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    for row in summary:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    run()
