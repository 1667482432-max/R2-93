from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def run() -> None:
    v3 = np.asarray([
        0.6495592946192368, 0.6427341038616939, 0.6128895349616729,
        0.6342139025915459, 0.5886195472315351,
    ])
    previous = np.asarray([
        0.6535324101099365, 0.6472264313905823, 0.6147747104138479,
        0.6388616795253408, 0.5936425945170071,
    ])
    validation = json.loads((ROOT / "matched_g6_validation.json").read_text(encoding="utf-8"))
    rows = [
        row for row in validation["records"]
        if row["profile"] == "g0_g1_g5_g6_g7_g8"
    ]
    rows.sort(key=lambda row: row["fold"])
    candidate = np.asarray([row["score"] for row in rows])
    result = {
        "candidate_scores": candidate.tolist(),
        "v3_scores": v3.tolist(),
        "previous_release_scores": previous.tolist(),
        "delta_vs_v3": (candidate - v3).tolist(),
        "mean_delta_vs_v3": float(np.mean(candidate - v3)),
        "delta_vs_previous_release": (candidate - previous).tolist(),
        "mean_delta_vs_previous_release": float(np.mean(candidate - previous)),
        "min_delta_vs_previous_release": float(np.min(candidate - previous)),
        "qualifies_next_0p004_milestone": bool(
            np.mean(candidate - previous) >= 0.004 and np.all(candidate > previous)
        ),
    }
    (ROOT / "matched_joint_milestone2_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
