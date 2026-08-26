from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
PROFILES = {
    "matched_core_nog10": {"R2_MATCHED_KERNEL_CORE_NOG10": "1"},
    "matched_core_nog10_safe": {
        "R2_MATCHED_KERNEL_CORE_NOG10": "1", "R2_MATCHED_TARGET_SAFE": "1",
        "R2_PAS_POOL": "1", "R2_PAS_SMOOTH": "1",
        "R2_PDP_POOL": "1", "R2_PDP_SMOOTH": "1",
        "R2_PDP_POOL_GROUPS": "",
    },
}
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
    "R2_MATCHED_TARGET_SAFE",
    "R2_PAS_POOL", "R2_PAS_SMOOTH", "R2_PDP_POOL", "R2_PDP_SMOOTH",
    "R2_PDP_POOL_GROUPS", "R2_SAVE_PRED",
}
COMMON = {
    "R2_AFFINE_PROFILE": "1", "R2_QUADRATIC_PROFILE": "1",
    "R2_PROJECTION_ITERS_PROFILE": "1", "R2_FINAL_PAS_PROFILE": "1",
    "R2_PAS_LOW_RANK": "1", "R2_PAS_MOMENT_ALIGN": "1",
    "R2_SHADOW_KERNEL_PROFILE": "1", "R2_SHADOW_CORRECTION_NOG7": "1",
    "R2_SHADOW_ITERATION_PROFILE_V3": "1", "R2_ROBUST_AMPLITUDE": "0.01",
    "R2_PDP_POOL": "1", "R2_PDP_POOL_GROUPS": "0,1,3,4,5,6,8,10",
}
BASELINE = [
    0.6495592946192368, 0.6427341038616939, 0.6128895349616729,
    0.6342139025915459, 0.5886195472315351,
]


def one(profile: str, fold: int) -> dict:
    env = os.environ.copy()
    for name in CLEAR:
        env.pop(name, None)
    env.update(COMMON)
    env.update(PROFILES[profile])
    env["R2_VAL_FILE"] = f"matched_rect_val_{fold}.npy"
    env["R2_VAL_GROUP_FILE"] = f"matched_rect_groups_{fold}.npy"
    if profile == "matched_core_nog10_safe":
        env["R2_SAVE_PRED"] = f"matched_pred_core_nog10_safe_fold{fold}.npy"
    result = subprocess.run(
        [str(PYTHON), "r2_pipeline.py", "island-validate"], cwd=ROOT,
        env=env, text=True, capture_output=True, check=True,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()
            if line.startswith("{")]
    weighted = next(row for row in rows
                    if row.get("method") == "island_official_weighted")
    islands = {int(row["island"]): row for row in rows if "island" in row}
    (ROOT / f"matched_candidate_{profile}_fold{fold}.out").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    print(json.dumps({"profile": profile, "fold": fold,
                      "score": weighted["score"],
                      "delta": weighted["score"] - BASELINE[fold]}), flush=True)
    return {"profile": profile, "fold": fold,
            "weighted": weighted, "islands": islands}


def run() -> None:
    output = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, profile, fold)
                   for profile in PROFILES for fold in range(5)]
        for future in concurrent.futures.as_completed(futures):
            output.append(future.result())
    output.sort(key=lambda row: (row["profile"], row["fold"]))
    (ROOT / "matched_candidate_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    summary = {}
    for profile in PROFILES:
        scores = [row["weighted"]["score"] for row in output
                  if row["profile"] == profile]
        delta = [score - base for score, base in zip(scores, BASELINE)]
        summary[profile] = {"scores": scores, "deltas": delta,
                            "mean": sum(scores) / 5,
                            "mean_delta": sum(delta) / 5,
                            "min_delta": min(delta)}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
