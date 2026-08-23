from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
from build_phase9_submission import (
    gate_features_test,
    interpolate_pas_residual,
    official_anchors,
)
from matched_phase5_tree_band_descriptor import build_cache
from matched_phase6_pas_fitted_canonical import direction, h_moment
from matched_phase6_pas_fitted_vertical import v_moment
from matched_phase7_pas_aggregate_canonical import aggregate
from matched_phase7_pas_aggregate_graph_metric import build_coordinates
from matched_phase7_pas_aggregate_portfolio_exact import PORTFOLIOS
from phase8_anchor_augmented_local_pas_screen import local_prediction
from phase8_anchor_local_gate_channel_validation import (
    ALPHA_GRID,
    components,
    prepare_folds,
)
from phase8_anchor_retained_pas_resolution_validation import project
from phase8_anchor_retained_pas_screen import (
    horizontal_shifts,
    mapped_anchors,
    normalize,
)
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_buildable_residual_gate_pas_screen import external_residual
from phase10_calibrated_anchor_on_pas_validation import calibrated_weights
from phase10_calibrated_pas_residual_joint_validation import update_weighted_scores
from phase10_pas_complement_predeclared_audit import mixed_target
from phase10_phase7_component_decompose import (
    action_weight_map,
    build_fold_action_logs,
    load_selections,
)
from phase10_robust125_primary_anchor_joint_validation import (
    build_robust_test_source,
    robust_weights,
    sha256,
    weights_manifest,
)


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
CORE_GROUPS = {1, 3, 4, 9, 10}
COMPLEMENT_GROUPS = set(range(11)) - CORE_GROUPS
CORE_ALPHA = 1.25
COMPLEMENT_ALPHA = 1.0
ANCHOR_GROUPS = (4, 5, 10)
RESIDUAL_ALPHA = 0.10
LOCAL_SCALE = 0.50
LOCAL_CLIP = 0.30
ITERATIONS = 4
FOLD_WEIGHTS = np.asarray([0.312, 0.357, 0.229, 0.046, 0.057], dtype=np.float64)
FOLD_WEIGHTS /= FOLD_WEIGHTS.sum()
LOCKED_FOLD_BY_GROUP = {
    0: 1,
    1: 1,
    2: 0,
    3: 1,
    4: 2,
    5: 0,
    6: 1,
    7: 1,
    8: 0,
    9: 0,
    10: 2,
}
VALIDATION_PATH = ROOT / "phase10_core125_complement100_primary_anchor_confirmation.json"
RAW_TEST_PATH = ROOT / "phase10_core125_complement100_pas_band24_test.npy"
FINAL_TEST_PATH = (
    ROOT / "phase10_core125_complement100_primary_anchor_pas_band24_test.npy"
)
MANIFEST_PATH = ROOT / "phase10_core125_complement100_primary_anchor_manifest.json"


def log(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}), flush=True)


def combined_weights() -> tuple[dict, dict]:
    core = robust_weights()
    global_selection, _ = load_selections()
    global_weights = action_weight_map(global_selection)
    joint = core.copy()
    for group in COMPLEMENT_GROUPS:
        for index, action in enumerate(PORTFOLIOS[group]):
            joint[(group, index)] = (
                global_weights[(group, index)]
                if action[0] in {"graph", "canonical", "gp"}
                else 0.0
            )
    return core, joint


def composite_components(
    group_accumulators: list[np.ndarray], index: int, mapping: dict[int, int]
) -> dict[str, float]:
    total = sum(
        (
            group_accumulators[fold][index, group]
            for group, fold in mapping.items()
        ),
        np.zeros(6, dtype=np.float64),
    )
    return components(total)


def apply_primary_anchor(
    target: np.ndarray,
    residual: np.ndarray,
    local: np.ndarray,
    labels: np.ndarray,
    raw_gate: np.ndarray,
) -> np.ndarray:
    group_mask = np.isin(labels, ANCHOR_GROUPS).astype(np.float32)[
        :, None, None, None
    ]
    corrected = normalize(
        target * np.exp(RESIDUAL_ALPHA * group_mask * residual)
    )
    alpha = group_mask * np.clip(
        LOCAL_SCALE * raw_gate, 0.0, LOCAL_CLIP
    )[:, None, None, None]
    return normalize((1.0 - alpha) * corrected + alpha * local).astype(np.float32)


def build_test_target(
    folds: list[dict],
    pos: np.ndarray,
    energy: np.ndarray,
    official_counts: dict[int, int],
    joint_weights: dict,
    validation_summary: dict,
) -> dict:
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    valid = np.flatnonzero(energy > 0)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    group_alphas = {
        group: CORE_ALPHA if group in CORE_GROUPS else COMPLEMENT_ALPHA
        for group in range(11)
    }
    raw_target, source_diagnostic = build_robust_test_source(
        pos,
        test_pos,
        valid,
        target,
        test_labels,
        joint_weights,
        group_alphas=group_alphas,
    )
    np.save(RAW_TEST_PATH, raw_target)
    log("raw_test_target_saved", path=RAW_TEST_PATH.name)

    anchors, anchor_labels, _, anchor_counts = official_anchors(
        pos, energy, test_pos
    )
    external_train = np.setdiff1d(valid, anchors)
    all_pos = np.vstack((pos, test_pos))
    test_index = np.arange(len(pos), len(all_pos))
    shifts = horizontal_shifts(all_pos)
    base = np.asarray(
        np.load(
            ROOT / "matched_phase6_milestone_physics_pas_band24_test.npy",
            mmap_mode="r",
        )
    )
    local = local_prediction(
        all_pos[:, :2], shifts, target, valid, test_index, 4, 3.0, "none"
    )
    anchor_base = local_prediction(
        all_pos[:, :2],
        shifts,
        target,
        external_train,
        anchors,
        4,
        3.0,
        "none",
    )
    epsilon = 1e-4 / 256
    anchor_log = np.clip(
        np.log((np.asarray(target[anchors]) + epsilon) / (anchor_base + epsilon)),
        -2.0,
        2.0,
    )
    anchor_log = np.repeat(
        anchor_log.mean(3, keepdims=True), anchor_log.shape[3], axis=3
    )
    residual = interpolate_pas_residual(
        pos, test_pos, test_labels, anchors, anchor_labels, anchor_log
    )

    gate_x = np.concatenate([row["x"] for row in folds])
    gate_y = np.concatenate([row["gain_grid"] for row in folds])
    gate_w = np.concatenate([row["calibrated_weights"] for row in folds])
    test_x = gate_features_test(
        pos, test_pos, valid, anchors, test_labels, base, local
    )
    gate = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=80,
        max_features=0.7,
        n_jobs=-1,
        random_state=52180,
    )
    gate.fit(gate_x, gate_y, sample_weight=gate_w)
    raw_gate = ALPHA_GRID[np.argmax(gate.predict(test_x), axis=1)]
    final_target = apply_primary_anchor(
        raw_target, residual, local, test_labels, raw_gate
    )
    np.save(FINAL_TEST_PATH, final_target)
    log("final_test_target_saved", path=FINAL_TEST_PATH.name)

    group_mask = np.isin(test_labels, ANCHOR_GROUPS)
    applied_alpha = np.zeros(len(test_pos), dtype=np.float32)
    applied_alpha[group_mask] = np.clip(
        LOCAL_SCALE * raw_gate[group_mask], 0.0, LOCAL_CLIP
    )
    manifest = {
        "validation_passed": True,
        "validation_gate": (
            "relative to core robust125 + primary anchor: locked edge increment "
            "> 0 and all five fold increments >= 0"
        ),
        "validation_summary": validation_summary,
        "parameters": {
            "core_groups": sorted(CORE_GROUPS),
            "core_family": "frozen robust_binary_all subset",
            "core_alpha": CORE_ALPHA,
            "complement_groups": sorted(COMPLEMENT_GROUPS),
            "complement_family": (
                "graph_canonical_gp with frozen global-selection weights"
            ),
            "complement_alpha": COMPLEMENT_ALPHA,
            "anchor_groups": list(ANCHOR_GROUPS),
            "residual_alpha": RESIDUAL_ALPHA,
            "local_scale": LOCAL_SCALE,
            "local_clip": LOCAL_CLIP,
            "projection_bands": 24,
            "projection_iterations": ITERATIONS,
        },
        "raw_pas_target": {
            "path": RAW_TEST_PATH.name,
            "shape": list(raw_target.shape),
            "dtype": str(raw_target.dtype),
            "bytes": RAW_TEST_PATH.stat().st_size,
            "sha256": sha256(RAW_TEST_PATH),
        },
        "final_joint_pas_target": {
            "path": FINAL_TEST_PATH.name,
            "shape": list(final_target.shape),
            "dtype": str(final_target.dtype),
            "bytes": FINAL_TEST_PATH.stat().st_size,
            "sha256": sha256(FINAL_TEST_PATH),
            "min": float(final_target.min()),
            "negative_fraction": float(np.mean(final_target < 0)),
        },
        "test_geometry": {
            "official_counts": {
                str(k): int(v) for k, v in official_counts.items()
            },
            "anchor_count": int(len(anchors)),
            "anchor_counts": {
                str(k): int(v) for k, v in anchor_counts.items()
            },
            "gate_raw_quantiles": np.quantile(
                raw_gate, [0.0, 0.1, 0.5, 0.9, 1.0]
            ).tolist(),
            "applied_alpha_mean_all": float(applied_alpha.mean()),
            "applied_alpha_mean_anchor_groups": float(
                applied_alpha[group_mask].mean()
            ),
        },
        "source_diagnostic": source_diagnostic,
        "actions": weights_manifest(joint_weights),
        "next_step": (
            "project Phase6 channel toward final_joint_pas_target with 24 bands "
            "and 4 iterations; preserve PDP target"
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("manifest_saved", path=MANIFEST_PATH.name)
    return manifest


@torch.no_grad()
def run() -> None:
    folds, pos, channel, energy, official_counts, actual_counts = prepare_folds()
    target = build_cache()
    aggregate_target = aggregate(target)
    valid_mask = energy > 0
    valid = np.flatnonzero(valid_mask)
    unit, side = direction(pos)
    horizontal_moment = h_moment(target, valid)
    vertical_moment = v_moment(target, valid)
    coordinates = build_coordinates(pos, valid)
    split_diagnostics = json.loads(
        (ROOT / "matched_rect_split_diagnostics.json").read_text()
    )
    core_weights, joint_weights = combined_weights()

    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    official_d1 = np.asarray(
        cKDTree(pos[valid, :2]).query(test_pos[:, :2], k=1)[0]
    )
    official_log_d1 = {
        int(group): float(
            np.mean(np.log1p(official_d1[test_labels == group]))
        )
        for group in np.unique(test_labels)
    }
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)
    phase8_shifts = horizontal_shifts(pos)
    calibration_diagnostics = []
    reconstruction_errors = []

    for fold, row in enumerate(folds):
        val, labels, old_base, action_logs, reconstruction_error = (
            build_fold_action_logs(
                fold,
                pos,
                energy,
                target,
                aggregate_target,
                unit,
                side,
                horizontal_moment,
                vertical_moment,
                coordinates,
                split_diagnostics,
            )
        )
        row["core_target"] = mixed_target(
            row["base"],
            labels,
            row["query_local"],
            action_logs,
            core_weights,
            joint_weights,
            0.0,
        )
        row["joint_target"] = mixed_target(
            row["base"],
            labels,
            row["query_local"],
            action_logs,
            core_weights,
            joint_weights,
            COMPLEMENT_ALPHA,
        )
        anchors = mapped_anchors(
            pos, val, labels, actual_fraction, official_counts
        )
        external_mask = valid_mask.copy()
        external_mask[val] = False
        external_train = np.flatnonzero(external_mask)
        row["residual"] = external_residual(
            pos,
            val,
            labels,
            anchors,
            row["query_local"],
            target,
            external_train,
            phase8_shifts,
            "none",
        )
        train_mask = external_mask.copy()
        train_mask[val[anchors]] = True
        row["calibrated_weights"], diagnostic = calibrated_weights(
            pos,
            row["query"],
            labels[row["query_local"]],
            np.flatnonzero(train_mask),
            official_counts,
            official_log_d1,
        )
        calibration_diagnostics.append(diagnostic)
        reconstruction_errors.append(reconstruction_error)
        log("prepare_fold", fold=fold, reconstruction_error=reconstruction_error)

    gate_alpha = []
    for heldout in range(5):
        gate_x = np.concatenate(
            [folds[index]["x"] for index in range(5) if index != heldout]
        )
        gate_y = np.concatenate(
            [folds[index]["gain_grid"] for index in range(5) if index != heldout]
        )
        gate_w = np.concatenate(
            [
                folds[index]["calibrated_weights"]
                for index in range(5)
                if index != heldout
            ]
        )
        gate = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=80,
            max_features=0.7,
            n_jobs=-1,
            random_state=52180,
        )
        gate.fit(gate_x, gate_y, sample_weight=gate_w)
        gate_alpha.append(
            ALPHA_GRID[np.argmax(gate.predict(folds[heldout]["x"]), axis=1)]
        )

    names = ("phase6", "core", "core_anchor", "joint", "joint_anchor")
    group_accumulators = []
    fold_rows = []
    for fold, row in enumerate(folds):
        labels = row["labels"][row["query_local"]]
        core_anchor = apply_primary_anchor(
            row["core_target"],
            row["residual"],
            row["local"],
            labels,
            gate_alpha[fold],
        )
        joint_anchor = apply_primary_anchor(
            row["joint_target"],
            row["residual"],
            row["local"],
            labels,
            gate_alpha[fold],
        )
        targets = {
            "core": row["core_target"],
            "core_anchor": core_anchor,
            "joint": row["joint_target"],
            "joint_anchor": joint_anchor,
        }
        prediction = np.load(
            ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r"
        )
        accumulator = np.zeros((len(names), 6), dtype=np.float64)
        group_accumulator = np.zeros((len(names), 11, 6), dtype=np.float64)
        for start in range(0, len(row["query"]), 8):
            stop = min(start + 8, len(row["query"]))
            local_query = row["query_local"][start:stop]
            p = torch.as_tensor(
                np.asarray(prediction[local_query]).copy(), device=DEVICE
            )
            t = torch.as_tensor(
                np.asarray(channel[row["query"][start:stop]]).copy(), device=DEVICE
            )
            batch_weights = torch.as_tensor(
                row["calibrated_weights"][start:stop].astype(np.float32),
                device=DEVICE,
            )
            truth_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            truth_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
            update_weighted_scores(
                accumulator,
                group_accumulator,
                0,
                p,
                t,
                truth_pas,
                truth_pdp,
                batch_weights,
                labels[start:stop],
            )
            base_band = torch.as_tensor(
                row["base"][start:stop].copy(), device=DEVICE
            )
            for index, name in enumerate(names[1:], 1):
                target_band = torch.as_tensor(
                    targets[name][start:stop].copy(), device=DEVICE
                )
                value = project(
                    p, base_band, target_band, 24, ITERATIONS
                )
                update_weighted_scores(
                    accumulator,
                    group_accumulator,
                    index,
                    value,
                    t,
                    truth_pas,
                    truth_pdp,
                    batch_weights,
                    labels[start:stop],
                )

        values = {name: components(accumulator[index]) for index, name in enumerate(names)}
        fold_rows.append(
            {
                "fold": fold,
                **values,
                "core_anchor_delta_vs_phase6": (
                    values["core_anchor"]["score"] - values["phase6"]["score"]
                ),
                "joint_anchor_delta_vs_phase6": (
                    values["joint_anchor"]["score"] - values["phase6"]["score"]
                ),
                "joint_anchor_increment_vs_core_anchor": (
                    values["joint_anchor"]["score"]
                    - values["core_anchor"]["score"]
                ),
            }
        )
        group_accumulators.append(group_accumulator)
        log("exact_fold", fold=fold, result=fold_rows[-1])

    mappings = {
        "locked": LOCKED_FOLD_BY_GROUP,
        **{
            f"inner_{rotation}": {
                group: (fold + rotation) % 5
                for group, fold in LOCKED_FOLD_BY_GROUP.items()
            }
            for rotation in range(1, 5)
        },
    }
    composites = {}
    for mapping_name, mapping in mappings.items():
        values = {
            name: composite_components(group_accumulators, index, mapping)
            for index, name in enumerate(names)
        }
        composites[mapping_name] = {
            **values,
            "core_anchor_delta_vs_phase6": (
                values["core_anchor"]["score"] - values["phase6"]["score"]
            ),
            "joint_anchor_delta_vs_phase6": (
                values["joint_anchor"]["score"] - values["phase6"]["score"]
            ),
            "joint_anchor_increment_vs_core_anchor": (
                values["joint_anchor"]["score"]
                - values["core_anchor"]["score"]
            ),
        }

    core_delta = np.asarray(
        [row["core_anchor_delta_vs_phase6"] for row in fold_rows]
    )
    joint_delta = np.asarray(
        [row["joint_anchor_delta_vs_phase6"] for row in fold_rows]
    )
    increment = joint_delta - core_delta
    rotation_increments = np.asarray(
        [
            composites[f"inner_{rotation}"][
                "joint_anchor_increment_vs_core_anchor"
            ]
            for rotation in range(1, 5)
        ]
    )
    summary = {
        "core_anchor_fold_deltas_vs_phase6": core_delta.tolist(),
        "joint_anchor_fold_deltas_vs_phase6": joint_delta.tolist(),
        "fold_increments_vs_core_anchor": increment.tolist(),
        "core_anchor_geometry_weighted_delta": float(
            np.dot(FOLD_WEIGHTS, core_delta)
        ),
        "joint_anchor_geometry_weighted_delta": float(
            np.dot(FOLD_WEIGHTS, joint_delta)
        ),
        "geometry_weighted_increment_vs_core_anchor": float(
            np.dot(FOLD_WEIGHTS, increment)
        ),
        "locked_edge_core_anchor_delta": composites["locked"][
            "core_anchor_delta_vs_phase6"
        ],
        "locked_edge_joint_anchor_delta": composites["locked"][
            "joint_anchor_delta_vs_phase6"
        ],
        "locked_edge_increment_vs_core_anchor": composites["locked"][
            "joint_anchor_increment_vs_core_anchor"
        ],
        "rotation_increments_vs_core_anchor": rotation_increments.tolist(),
        "rotation_positive_count": int(np.sum(rotation_increments > 0)),
        "all_fold_non_decreasing": bool(np.all(increment >= 0.0)),
    }
    passed = bool(
        summary["locked_edge_increment_vs_core_anchor"] > 0.0
        and summary["all_fold_non_decreasing"]
    )
    summary["fixed_confirmation_passed"] = passed
    output = {
        "frozen_before_evaluation": True,
        "selection_or_tuning_performed": False,
        "parameters": {
            "core_groups": sorted(CORE_GROUPS),
            "core_alpha": CORE_ALPHA,
            "complement_groups": sorted(COMPLEMENT_GROUPS),
            "complement_family": "graph_canonical_gp frozen global weights",
            "complement_alpha": COMPLEMENT_ALPHA,
            "anchor_groups": list(ANCHOR_GROUPS),
            "residual_alpha": RESIDUAL_ALPHA,
            "local_scale": LOCAL_SCALE,
            "local_clip": LOCAL_CLIP,
            "projection_bands": 24,
            "projection_iterations": ITERATIONS,
        },
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "locked_fold_by_group": LOCKED_FOLD_BY_GROUP,
        "official_log_d1": official_log_d1,
        "reconstruction_max_errors": reconstruction_errors,
        "calibration_diagnostics": calibration_diagnostics,
        "core_actions": weights_manifest(core_weights),
        "joint_actions": weights_manifest(joint_weights),
        "folds": fold_rows,
        "composites": composites,
        "summary": summary,
    }
    VALIDATION_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log("validation_saved", path=VALIDATION_PATH.name, summary=summary)

    if passed:
        manifest = build_test_target(
            folds, pos, energy, official_counts, joint_weights, summary
        )
        output["test_target_manifest"] = manifest
        VALIDATION_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    else:
        log("test_target_skipped", reason="fixed confirmation gate not met")


if __name__ == "__main__":
    run()
