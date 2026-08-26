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
    "v3_official_best": {
        "R2_SHADOW_CORRECTION_NOG7": "1",
        "R2_SHADOW_ITERATION_PROFILE_V3": "1",
    },
    "v4_pooling": {
        "R2_SHADOW_CORRECTION_NOG7": "1",
        "R2_SHADOW_ITERATION_PROFILE_V3": "1",
        "R2_SHADOW_TARGET_PROFILE_V4": "1",
        "R2_SHADOW_TARGET_PROFILE_V4_G7": "1",
        "R2_PAS_POOL": "1", "R2_PAS_SMOOTH": "1",
        "R2_PDP_POOL": "1", "R2_PDP_SMOOTH": "1",
        "R2_PDP_POOL_GROUPS": "",
    },
    "v5_rejected": {
        "R2_SHADOW_CORRECTION_NOG7": "1",
        "R2_SHADOW_ITERATION_PROFILE_V3": "1",
        "R2_SHADOW_TARGET_PROFILE_V4": "1",
        "R2_SHADOW_TARGET_PROFILE_V4_G7": "1",
        "R2_SHADOW_PAS2D_PROFILE_V5": "1",
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
    "R2_PDP_POOL_GROUPS", "R2_SHADOW_TARGET_PROFILE_V4",
    "R2_SHADOW_TARGET_PROFILE_V4_G7", "R2_SHADOW_PAS2D_PROFILE_V5",
    "R2_SHADOW_PAS2D_PROFILE_V5_EXTENDED", "R2_PAS_POOL",
    "R2_PAS_SMOOTH", "R2_PDP_POOL", "R2_PDP_SMOOTH", "R2_SAVE_PRED",
}

COMMON = {
    "R2_AFFINE_PROFILE": "1", "R2_QUADRATIC_PROFILE": "1",
    "R2_PROJECTION_ITERS_PROFILE": "1", "R2_FINAL_PAS_PROFILE": "1",
    "R2_PAS_LOW_RANK": "1", "R2_PAS_MOMENT_ALIGN": "1",
    "R2_SHADOW_KERNEL_PROFILE": "1", "R2_ROBUST_AMPLITUDE": "0.01",
    # V3 retains the conservative PDP sharing used by its submitted artifact.
    "R2_PDP_POOL": "1", "R2_PDP_POOL_GROUPS": "0,1,3,4,5,6,8,10",
}


def one(profile: str, fold: int) -> dict:
    env = os.environ.copy()
    for name in CLEAR:
        env.pop(name, None)
    env.update(COMMON)
    env.update(PROFILES[profile])
    env["R2_VAL_FILE"] = f"matched_rect_val_{fold}.npy"
    env["R2_VAL_GROUP_FILE"] = f"matched_rect_groups_{fold}.npy"
    if profile in {"v3_official_best", "v5_rejected"}:
        env["R2_SAVE_PRED"] = f"matched_pred_{profile}_fold{fold}.npy"
    result = subprocess.run(
        [str(PYTHON), "r2_pipeline.py", "island-validate"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    records = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith("{")]
    weighted = next(item for item in records
                    if item.get("method") == "island_official_weighted")
    islands = {int(item["island"]): item for item in records if "island" in item}
    (ROOT / f"matched_validation_{profile}_fold{fold}.out").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    print(json.dumps({"profile": profile, "fold": fold,
                      "score": weighted["score"]}), flush=True)
    return {"profile": profile, "fold": fold,
            "weighted": weighted, "islands": islands}


def run() -> None:
    output = []
    jobs = [(profile, fold) for profile in PROFILES for fold in range(5)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            output.append(future.result())
    output.sort(key=lambda item: (item["profile"], item["fold"]))
    (ROOT / "matched_rank_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    summary = {}
    for profile in PROFILES:
        scores = [item["weighted"]["score"] for item in output
                  if item["profile"] == profile]
        summary[profile] = {"scores": scores, "mean": sum(scores) / len(scores)}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
