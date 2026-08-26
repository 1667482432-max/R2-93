from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from matched_phase6_groupwise_physics_validation import build as build_groupwise


ROOT = Path(__file__).resolve().parent
RULES = {
    0: (11, 6.5402, "low", 0.30),
    3: (2, 2.9594, "low", 0.30),
    5: (13, 18.1151, "low", 0.20),
    7: (15, 43.645328521728516, "high", 0.75),
    10: (10, 6.0715, "low", 0.05),
}


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def build() -> None:
    build_groupwise()
    gate = np.load(ROOT / "matched_phase6_pas_rich_gate_features.npz")
    for fold in range(5):
        desired = np.load(
            ROOT / f"matched_phase6_groupwise_physics_pas_band24_fold{fold}.npy"
        ).copy()
        rich = np.load(
            ROOT / f"matched_rich_tree_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        labels = gate[f"labels{fold}"]
        features = gate[f"x{fold}"]
        selected = {}
        for group, (feature, threshold, direction, alpha) in RULES.items():
            mask = labels == group
            mask &= (
                features[:, feature] <= threshold
                if direction == "low"
                else features[:, feature] >= threshold
            )
            desired[mask] = normalize((1.0 - alpha) * desired[mask] + alpha * rich[mask])
            selected[str(group)] = int(mask.sum())
        np.save(
            ROOT / f"matched_phase6_enhanced_physics_pas_band24_fold{fold}.npy",
            desired.astype(np.float32),
        )
        print(json.dumps({"stage": "enhanced_descriptor", "fold": fold, "selected": selected}), flush=True)


def run() -> None:
    build()
    os.environ["R2_PHYSICS_DESCRIPTOR_LABEL"] = "enhanced_physics"
    from matched_phase6_physics_combo_channel_validation import run as validate

    validate()


if __name__ == "__main__":
    run()
