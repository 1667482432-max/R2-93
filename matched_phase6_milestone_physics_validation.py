from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from matched_phase6_enhanced_physics_validation import build as build_enhanced


ROOT = Path(__file__).resolve().parent
RULES = {
    0: ("matched_fitted_canonical_leaf2_mf8_pas_band24_fold{}.npy", 9, -0.6488235473632791, "high", 0.20),
    1: ("matched_fitted_vertical_pas_band24_fold{}.npy", 6, 18.14078559875488, "low", 0.15),
    5: ("matched_tree_pas_canonical_c1_k5p0_fold{}.npy", 0, 56.46230506896973, "low", 0.075),
    6: ("matched_fitted_canonical_leaf3_mf5_pas_band24_fold{}.npy", 8, 0.14137563109397888, "high", 1.00),
    7: ("matched_tree_pas_canonical_c1_k5p0_fold{}.npy", 15, 43.645328521728516, "high", 0.40),
    9: ("matched_fitted_vertical_pas_band24_fold{}.npy", 10, 6.596559524536133, "high", 0.10),
    10: ("matched_rich_mlp_pas_band24_fold{}.npy", 8, -2.2051992416381836, "low", 0.50),
}


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def build() -> None:
    build_enhanced()
    gate = np.load(ROOT / "matched_phase6_pas_rich_gate_features.npz")
    for fold in range(5):
        desired = np.load(
            ROOT / f"matched_phase6_enhanced_physics_pas_band24_fold{fold}.npy"
        ).copy()
        labels = gate[f"labels{fold}"]
        features = gate[f"x{fold}"]
        selected = {}
        for group, (pattern, feature, threshold, direction, alpha) in RULES.items():
            candidate = np.load(ROOT / pattern.format(fold), mmap_mode="r")
            mask = labels == group
            mask &= (
                features[:, feature] <= threshold
                if direction == "low"
                else features[:, feature] >= threshold
            )
            desired[mask] = normalize((1.0 - alpha) * desired[mask] + alpha * candidate[mask])
            selected[str(group)] = int(mask.sum())
        np.save(
            ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
            desired.astype(np.float32),
        )
        print(json.dumps({"stage": "milestone_descriptor", "fold": fold, "selected": selected}), flush=True)


def run() -> None:
    build()
    os.environ["R2_PHYSICS_DESCRIPTOR_LABEL"] = "milestone_physics"
    from matched_phase6_physics_combo_channel_validation import run as validate

    validate()


if __name__ == "__main__":
    run()
