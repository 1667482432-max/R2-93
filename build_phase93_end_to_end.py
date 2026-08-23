"""Rebuild the frozen Phase93 chain from user-supplied competition data.

This orchestrator never downloads or embeds competition data.  It runs the
published P9 builder, the frozen Phase10 PAS projection, Phase40 anti-P10
composition, and finally Phase93 in that order.  Every stage is fail-closed
and refuses to overwrite an existing output.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMP = Path(os.environ["TEMP"])
RAW = ("Round2_Train_Channel.npy", "Round2_Train_Pos.npy", "Round2_Test_Pos.npy", "train_energy.npy")
DERIVED = (
    "Round2_Test_Channel_matched_phase6_delta2053.npy",
    "pas_ue_band24_descriptors.npy",
    "matched_phase6_milestone_physics_pas_band24_test.npy",
    "phase10_core125_complement100_primary_anchor_pas_band24_test.npy",
    "los_map_features.npy",
    "rich_map_features.npy",
    "matched_tree_pas_band24_test.npy",
    "matched_rbf_pas_band24_test.npy",
    "matched_phase6_pas_rich_gate_features.npz",
)
OUTPUTS = (
    "Round2_Test_Channel_matched_phase9_buildable_anchor_joint.npy",
    "Round2_Test_Channel_phase10_core125_complement100_anchor_pas.npy",
    TEMP / "Round2_Test_Channel_phase40_p9_actual_antip10_pas050.npy",
    TEMP / "Round2_Test_Channel_phase93_g56_antip10_plus_symmetric_clamp.npy",
)


def run(script: str, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print(f"[phase93-e2e] running {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=merged, check=True)


def main() -> None:
    missing = [name for name in RAW if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError("Missing authorized competition data: " + ", ".join(missing))
    missing = [name for name in DERIVED if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError("Missing frozen derived prerequisite: " + ", ".join(missing))
    present = [str(ROOT / p) if isinstance(p, str) else str(p) for p in OUTPUTS if Path(ROOT / p if isinstance(p, str) else p).exists()]
    if present:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(present))
    run("build_phase9_submission.py")
    run(
        "build_phase10_robust125_anchor_pas_submission.py",
        {
            "R2_PHASE10_TARGET": "phase10_core125_complement100_primary_anchor_pas_band24_test.npy",
            "R2_PHASE10_VALIDATION": "phase10_core125_complement100_primary_anchor_confirmation.json",
            "R2_PHASE10_TARGET_MANIFEST": "phase10_core125_complement100_primary_anchor_manifest.json",
            "R2_PHASE10_OUTPUT": "Round2_Test_Channel_phase10_core125_complement100_anchor_pas.npy",
            "R2_PHASE10_MANIFEST": "phase10_core125_complement100_anchor_pas_submission_manifest.json",
        },
    )
    run("build_phase40_p9_actual_antip10_pas050_submission.py")
    run("build_phase93_g56_antip10_plus_symmetric_clamp_submission.py")
    print("[phase93-e2e] complete", flush=True)


if __name__ == "__main__":
    main()
