from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import r2_pipeline as rp
from run_matched_metric_validation import HybridView


ROOT = Path(__file__).resolve().parent


def run() -> None:
    _, channel, _ = rp.load_data()
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    milestone = json.loads(
        (ROOT / "matched_joint_milestone2_validation.json").read_text(encoding="utf-8")
    )["candidate_scores"]
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy")
        weights = np.asarray(
            [counts.get(int(group), 0) / max(1, np.sum(labels == group)) for group in labels]
        )
        base = np.load(ROOT / f"matched_phase2_full_fold{fold}.npy", mmap_mode="r")
        replacements = {
            1: np.load(
                ROOT / f"matched_refine_g1_affine_pas025_010_combo_fold{fold}.npy",
                mmap_mode="r",
            ),
            5: np.load(ROOT / f"matched_refine_g5_pas_smooth_hv010_fold{fold}.npy", mmap_mode="r"),
        }
        phase2 = rp.score_numpy_weighted(base, channel[val], weights)
        phase3 = rp.score_numpy_weighted(
            HybridView(base, labels, replacements), channel[val], weights
        )
        record = {
            "fold": fold,
            "milestone": milestone[fold],
            "phase2": phase2,
            "phase3": phase3,
            "delta_vs_phase2": phase3["score"] - phase2["score"],
            "delta_vs_milestone": phase3["score"] - milestone[fold],
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    summary = {
        "phase3_scores": [row["phase3"]["score"] for row in records],
        "deltas_vs_phase2": [row["delta_vs_phase2"] for row in records],
        "deltas_vs_milestone": [row["delta_vs_milestone"] for row in records],
    }
    summary["mean_delta_vs_phase2"] = float(np.mean(summary["deltas_vs_phase2"]))
    summary["min_delta_vs_phase2"] = float(np.min(summary["deltas_vs_phase2"]))
    summary["mean_delta_vs_milestone"] = float(np.mean(summary["deltas_vs_milestone"]))
    (ROOT / "matched_phase3_validation.json").write_text(
        json.dumps({"records": records, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
