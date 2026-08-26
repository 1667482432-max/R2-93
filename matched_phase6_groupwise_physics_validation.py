from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    1: (0.05, 0.15, 0.50),
    3: (0.00, 0.00, 1.00),
    4: (0.15, 0.00, 1.50),
    5: (0.00, 0.00, 1.00),
    7: (0.05, 0.10, 2.00),
    9: (0.00, 0.025, 1.50),
}


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def build() -> None:
    gate = np.load(ROOT / "matched_phase6_pas_rich_gate_features.npz")
    for fold in range(5):
        path = ROOT / f"matched_phase6_groupwise_physics_pas_band24_fold{fold}.npy"
        labels = gate[f"labels{fold}"]
        base = np.load(ROOT / f"matched_phase5_pas_band24_fold{fold}.npy", mmap_mode="r")
        horizontal = np.load(
            ROOT / f"matched_fitted_canonical_leaf8_mf8_pas_band24_fold{fold}.npy",
            mmap_mode="r",
        )
        vertical = np.load(
            ROOT / f"matched_fitted_vertical_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        safe = np.load(
            ROOT / f"matched_phase6_safe_combo_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        rich = np.load(
            ROOT / f"matched_rich_tree_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        desired = np.asarray(base).copy()
        for group, (horizontal_alpha, vertical_alpha, safe_scale) in CONFIGS.items():
            mask = labels == group
            desired[mask] = normalize(
                (1.0 - horizontal_alpha - vertical_alpha - safe_scale) * base[mask]
                + horizontal_alpha * horizontal[mask]
                + vertical_alpha * vertical[mask]
                + safe_scale * safe[mask]
            )
        group6 = (labels == 6) & (gate[f"x{fold}"][:, 8] > 1.3770753145217889)
        desired[group6] = normalize(0.5 * desired[group6] + 0.5 * rich[group6])
        np.save(path, desired.astype(np.float32))
        print(
            json.dumps({"stage": "descriptor", "fold": fold, "group6_rows": int(group6.sum())}),
            flush=True,
        )


def run() -> None:
    build()
    os.environ["R2_PHYSICS_DESCRIPTOR_LABEL"] = "groupwise_physics"
    from matched_phase6_physics_combo_channel_validation import run as validate

    validate()


if __name__ == "__main__":
    run()
