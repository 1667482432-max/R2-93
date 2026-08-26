from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp
from matched_local_spectral_calibration import (
    apply_per_sample,
    predict_profiles,
    residual_profiles,
)
from matched_spectral_calibration import (
    DEVICE,
    apply_correction,
    ratio_profile,
    spectral_sums,
)
from run_matched_metric_validation import HybridView


ROOT = Path(__file__).resolve().parent
GLOBAL_CONFIGS = {
    4: ("global", 0.50, 0.00, "pas_pdp"),
    5: ("global", 0.25, 0.00, "pas_pdp"),
    6: ("global", 0.25, 0.25, "pas_pdp"),
    7: ("global", 0.25, 0.25, "pas_pdp"),
    8: ("ue", 0.25, 0.00, "pas_pdp"),
}
LOCAL_CONFIGS = {
    0: (20, 0.20),
    3: (10, 0.20),
}


def align(candidate: np.ndarray, base: np.ndarray) -> np.ndarray:
    cross = np.sum(np.conj(candidate) * base, axis=(1, 2, 3), dtype=np.complex128)
    phase = cross / np.maximum(np.abs(cross), 1e-30)
    return candidate * phase[:, None, None, None].astype(np.complex64)


def g10_combo(base: np.ndarray, fold: int) -> np.ndarray:
    affine = align(
        np.asarray(np.load(
            ROOT / f"matched_refine_g10_affine_pas075_010_fold{fold}.npy",
            mmap_mode="r",
        )),
        base,
    )
    gradient = align(
        np.asarray(np.load(
            ROOT / f"matched_refine_g10_refine10_lr0003_fold{fold}.npy",
            mmap_mode="r",
        )),
        base,
    )
    return (base + (affine - base) + 0.10 * (gradient - base)).astype(np.complex64)


def run() -> None:
    pos, channel, _ = rp.load_data()
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    milestone = json.loads(
        (ROOT / "matched_joint_milestone2_validation.json").read_text(encoding="utf-8")
    )["candidate_scores"]
    phase3 = json.loads(
        (ROOT / "matched_phase3_validation.json").read_text(encoding="utf-8")
    )["summary"]["phase3_scores"]

    folds = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        base = np.load(ROOT / f"matched_phase2_full_fold{fold}.npy", mmap_mode="r")
        folds.append({
            "fold": fold,
            "val": val,
            "labels": labels,
            "base": base,
            "target": channel[val],
            "xy": pos[val, :2],
        })

    global_sums = {}
    for row in folds:
        for group in GLOBAL_CONFIGS:
            mask = row["labels"] == group
            global_sums[row["fold"], group] = spectral_sums(
                row["base"][mask], row["target"][mask]
            )
        print(json.dumps({"stage": "global_sums", "fold": row["fold"]}), flush=True)

    local_profiles = {}
    for row in folds:
        for group in LOCAL_CONFIGS:
            mask = row["labels"] == group
            local_profiles[row["fold"], group] = residual_profiles(
                row["base"][mask], row["target"][mask]
            )
        print(json.dumps({"stage": "local_profiles", "fold": row["fold"]}), flush=True)

    records = []
    for row in folds:
        fold = row["fold"]
        labels = row["labels"]
        base = row["base"]
        replacements: dict[int, np.ndarray] = {
            1: np.asarray(np.load(
                ROOT / f"matched_refine_g1_affine_pas025_010_combo_fold{fold}.npy",
                mmap_mode="r",
            )),
            5: np.asarray(np.load(
                ROOT / f"matched_refine_g5_pas_smooth_hv010_fold{fold}.npy",
                mmap_mode="r",
            )),
            7: np.asarray(np.load(
                ROOT / f"matched_refine_g7_pdp_smooth_d5020_fold{fold}.npy",
                mmap_mode="r",
            )),
        }
        rows10 = np.flatnonzero(labels == 10)
        replacements[10] = g10_combo(np.asarray(base[rows10]), fold)

        for group, (level, pas_blend, pdp_blend, order) in GLOBAL_CONFIGS.items():
            mask = labels == group
            source = replacements.get(group)
            pred = np.asarray(source if source is not None else base[mask])
            train_sums = [
                global_sums[other, group] for other in range(5) if other != fold
            ]
            pas_ratio = ratio_profile(train_sums, "pas", level)
            pdp_ratio = ratio_profile(train_sums, "pdp", level)
            tensor = torch.as_tensor(pred.copy(), device=DEVICE)
            replacements[group] = apply_correction(
                tensor, pas_ratio, pdp_ratio, pas_blend, pdp_blend, order
            ).cpu().numpy().astype(np.complex64)

        for group, (neighbors, blend) in LOCAL_CONFIGS.items():
            mask = labels == group
            train_rows = [other for other in folds if other["fold"] != fold]
            train_xy = np.concatenate([
                other["xy"][other["labels"] == group] for other in train_rows
            ])
            train_pas = np.concatenate([
                local_profiles[other["fold"], group][0] for other in train_rows
            ])
            train_pdp = np.concatenate([
                local_profiles[other["fold"], group][1] for other in train_rows
            ])
            pas_ratio, pdp_ratio = predict_profiles(
                train_xy, train_pas, train_pdp, row["xy"][mask], neighbors
            )
            tensor = torch.as_tensor(np.asarray(base[mask]).copy(), device=DEVICE)
            replacements[group] = apply_per_sample(
                tensor, pas_ratio, pdp_ratio, blend
            ).cpu().numpy().astype(np.complex64)

        if os.environ.get("R2_SAVE_PHASE4_FOLDS"):
            materialized = np.asarray(base).copy()
            for group, replacement in replacements.items():
                materialized[labels == group] = replacement
            np.save(ROOT / f"matched_phase4_full_fold{fold}.npy", materialized)
            del materialized

        weights = np.asarray([
            counts.get(int(group), 0) / max(1, np.sum(labels == group))
            for group in labels
        ])
        result = rp.score_numpy_weighted(
            HybridView(base, labels, replacements), row["target"], weights
        )
        record = {
            "fold": fold,
            "milestone": milestone[fold],
            "phase3": phase3[fold],
            "phase4": result,
            "delta_vs_milestone": result["score"] - milestone[fold],
            "delta_vs_phase3": result["score"] - phase3[fold],
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    deltas = [row["delta_vs_milestone"] for row in records]
    summary = {
        "scores": [row["phase4"]["score"] for row in records],
        "deltas_vs_milestone": deltas,
        "deltas_vs_phase3": [row["delta_vs_phase3"] for row in records],
        "mean_delta_vs_milestone": float(np.mean(deltas)),
        "min_delta_vs_milestone": float(np.min(deltas)),
        "qualifies_next_release": bool(np.mean(deltas) >= 0.004 and np.min(deltas) > 0),
    }
    (ROOT / "matched_phase4_composite_validation.json").write_text(
        json.dumps({"records": records, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
