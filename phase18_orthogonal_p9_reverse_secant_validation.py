from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
from phase8_anchor_augmented_local_pas_screen import local_prediction
from phase8_anchor_local_gate_channel_validation import ALPHA_GRID, components, prepare_folds
from phase8_anchor_retained_pas_screen import horizontal_shifts, mapped_anchors, normalize
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_anchor_joint_channel_validation import normalize_last, project_joint
from phase9_buildable_joint_channel_validation import external_pdp_residual
from phase9_buildable_residual_gate_pas_screen import external_residual
from phase10_calibrated_anchor_on_pas_validation import calibrated_weights
from phase10_calibrated_pas_residual_joint_validation import update_weighted_scores
from phase10_pas_complement_anchor_confirmation import (
    COMPLEMENT_ALPHA,
    FOLD_WEIGHTS,
    LOCKED_FOLD_BY_GROUP,
    apply_primary_anchor,
    combined_weights,
    composite_components,
)
from phase10_pas_complement_predeclared_audit import mixed_target
from phase10_phase7_component_decompose import build_fold_action_logs
from matched_phase6_pas_fitted_canonical import direction, h_moment
from matched_phase6_pas_fitted_vertical import v_moment
from matched_phase7_pas_aggregate_canonical import aggregate
from matched_phase7_pas_aggregate_graph_metric import build_coordinates


ROOT = Path(__file__).resolve().parent
PREDECLARED = ROOT / "phase18_orthogonal_p9_reverse_secant_predeclared.json"
OUTPUT = ROOT / "phase18_orthogonal_p9_reverse_secant_validation.json"
DEVICE = torch.device("cuda")
ETA = 1.0
PAS_EPSILON = 1e-4 / 256
PDP_EPSILON = 1e-4 / 192
PINV_RCOND = 1e-12
P9_RESIDUAL_ALPHA = 0.15
P9_LOCAL_SCALE = 0.75
P9_PDP_ALPHA = 0.025
ITERATIONS = 12


def log(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}), flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_predeclared() -> dict[str, Any]:
    value = json.loads(PREDECLARED.read_text(encoding="utf-8"))
    if (
        value["protocol"]
        != "phase18 orthogonalized P9 reverse secant eta1 exact fivefold veto"
        or value["frozen_before_validation"] is not True
        or value["scope"]["test_channel_materialization_allowed"] is not False
        or value["scope"]["upload_allowed"] is not False
        or float(value["official_feedback_used_only_for_direction_sign_and_eta"]["eta"])
        != ETA
    ):
        raise RuntimeError("Phase18 predeclaration does not match the validation code")
    return value


def normalize_pas(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def weighted_inner(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    # Keep the row tensors in float32 and only accumulate the per-row mean and
    # final weighted sum in float64.  Casting both full operands to float64 can
    # transiently triple memory during the fivefold audit without changing the
    # declared inner product in any meaningful way.
    product = np.asarray(left, dtype=np.float32) * np.asarray(right, dtype=np.float32)
    per_row = np.mean(
        product,
        axis=(1, 2, 3),
        dtype=np.float64,
    )
    return float(np.dot(np.asarray(weights, dtype=np.float64), per_row) / np.sum(weights))


def orthogonal_bad_residual(
    u65: np.ndarray,
    u8: np.ndarray,
    u9: np.ndarray,
    u10: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    basis = (u65, u8, u9)
    gram = np.asarray(
        [[weighted_inner(left, right, weights) for right in basis] for left in basis],
        dtype=np.float64,
    )
    cross = np.asarray(
        [weighted_inner(value, u10, weights) for value in basis], dtype=np.float64
    )
    coefficient = np.linalg.pinv(gram, rcond=PINV_RCOND, hermitian=True) @ cross
    residual = np.asarray(u10, dtype=np.float64).copy()
    for scale, value in zip(coefficient, basis):
        residual -= float(scale) * np.asarray(value, dtype=np.float64)
    norm10 = max(weighted_inner(u10, u10, weights), 1e-30)
    residual_norm = weighted_inner(residual, residual, weights)
    orthogonality = [
        weighted_inner(residual, value, weights)
        / np.sqrt(max(residual_norm * weighted_inner(value, value, weights), 1e-30))
        for value in basis
    ]
    return residual.astype(np.float32), {
        "gram": gram.tolist(),
        "cross": cross.tolist(),
        "projection_coefficients_u65_u8_u9": coefficient.tolist(),
        "bad_residual_norm_fraction": float(np.sqrt(max(residual_norm, 0.0) / norm10)),
        "residual_basis_cosines": [float(value) for value in orthogonality],
    }


def secant_targets(
    base_pas: np.ndarray,
    phase5_pas: np.ndarray,
    phase8_pas: np.ndarray,
    phase9_pas: np.ndarray,
    phase10_pas: np.ndarray,
    base_pdp: np.ndarray,
    phase9_pdp: np.ndarray,
    weights: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    u65 = np.log((np.asarray(base_pas) + PAS_EPSILON) / (phase5_pas + PAS_EPSILON))
    u8 = np.log((phase8_pas + PAS_EPSILON) / (base_pas + PAS_EPSILON))
    u9 = np.log((phase9_pas + PAS_EPSILON) / (base_pas + PAS_EPSILON))
    u10 = np.log((phase10_pas + PAS_EPSILON) / (base_pas + PAS_EPSILON))
    bad, diagnostic = orthogonal_bad_residual(u65, u8, u9, u10, weights)
    increment = u9 - bad
    raw_pas = np.asarray(phase9_pas, dtype=np.float64) * np.exp(
        eta * np.asarray(increment, dtype=np.float64)
    )
    target_pas = normalize_pas(raw_pas).astype(np.float32)
    v9 = np.log((phase9_pdp + PDP_EPSILON) / (base_pdp + PDP_EPSILON))
    raw_pdp = np.asarray(phase9_pdp, dtype=np.float64) * np.exp(
        eta * np.asarray(v9, dtype=np.float64)
    )
    target_pdp = normalize_last(raw_pdp).astype(np.float32)
    pas_ratio = (target_pas + 1e-3 / 24) / (base_pas + 1e-3 / 24)
    pdp_ratio = (target_pdp + PDP_EPSILON) / (base_pdp + PDP_EPSILON)
    eta0_pas = normalize_pas(np.asarray(phase9_pas, dtype=np.float64)).astype(np.float32)
    eta0_pdp = normalize_last(np.asarray(phase9_pdp, dtype=np.float64)).astype(np.float32)
    diagnostic.update(
        {
            "eta": eta,
            "pas_eta0_max_abs_target_error": float(np.max(np.abs(eta0_pas - phase9_pas))),
            "pdp_eta0_max_abs_target_error": float(np.max(np.abs(eta0_pdp - phase9_pdp))),
            "pas_projection_ratio_clip_fraction": float(
                np.mean((pas_ratio < 0.25) | (pas_ratio > 4.0))
            ),
            "pdp_projection_ratio_clip_fraction": float(
                np.mean((pdp_ratio < 0.25) | (pdp_ratio > 4.0))
            ),
            "pas_increment_abs_quantiles": np.quantile(
                np.abs(increment), [0.5, 0.9, 0.99, 0.999, 1.0]
            ).tolist(),
            "pdp_log_residual_abs_quantiles": np.quantile(
                np.abs(v9), [0.5, 0.9, 0.99, 0.999, 1.0]
            ).tolist(),
        }
    )
    return target_pas, target_pdp, diagnostic


def selected_phase9_reference() -> list[dict[str, float]]:
    data = json.loads(
        (ROOT / "phase9_buildable_joint_channel_validation.json").read_text(
            encoding="utf-8"
        )
    )
    output: list[dict[str, float]] = []
    for fold in data["folds"]:
        matches = [
            row
            for row in fold["rows"]
            if row["pas_label"] == "robust"
            and np.isclose(float(row["pdp_alpha"]), P9_PDP_ALPHA)
        ]
        if len(matches) != 1:
            raise RuntimeError("Cannot identify the exact Phase9 fold reference")
        output.append(
            {name: float(matches[0][name]) for name in ("c1_pas", "c2_pdp", "c3_nmse", "score")}
        )
    return output


def prepare_all_targets() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict, dict]:
    folds, pos, channel, energy, official_counts, actual_counts = prepare_folds()
    # The Phase18 basis is frozen at 24 PAS bands.  Loading the explicit cache
    # avoids inheriting the import-time R2_BANDS default (12) from older code.
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    if target.shape[1:] != (256, 4, 24):
        raise RuntimeError(f"Unexpected 24-band descriptor shape: {target.shape}")
    aggregate_target = aggregate(target)
    valid_mask = energy > 0
    valid = np.flatnonzero(valid_mask)
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)
    shifts = horizontal_shifts(pos)

    unit, side = direction(pos)
    horizontal_moment = h_moment(target, valid)
    vertical_moment = v_moment(target, valid)
    coordinates = build_coordinates(pos, valid)
    split_diagnostics = json.loads(
        (ROOT / "matched_rect_split_diagnostics.json").read_text(encoding="utf-8")
    )
    core_weights, joint_weights = combined_weights()
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    official_d1 = np.asarray(cKDTree(pos[valid, :2]).query(test_pos[:, :2], k=1)[0])
    official_log_d1 = {
        int(group): float(np.mean(np.log1p(official_d1[test_labels == group])))
        for group in np.unique(test_labels)
    }

    for fold, row in enumerate(folds):
        anchors = mapped_anchors(
            pos, row["val"], row["labels"], actual_fraction, official_counts
        )
        external_mask = valid_mask.copy()
        external_mask[row["val"]] = False
        external_train = np.flatnonzero(external_mask)
        row["pas_residual"] = external_residual(
            pos,
            row["val"],
            row["labels"],
            anchors,
            row["query_local"],
            target,
            external_train,
            shifts,
            "none",
        )
        base_pdp_all = np.load(
            ROOT / f"phase9_anchor_fold{fold}_base_pdp_band192.npy", mmap_mode="r"
        )
        truth_pdp_all = np.load(
            ROOT / f"phase9_anchor_fold{fold}_truth_pdp_band192.npy", mmap_mode="r"
        )
        row["base_pdp"] = np.asarray(base_pdp_all[row["query_local"]])
        row["pdp_residual"] = external_pdp_residual(
            pos,
            channel,
            row["val"],
            row["labels"],
            anchors,
            row["query_local"],
            external_train,
            truth_pdp_all,
        )
        train_mask = external_mask.copy()
        train_mask[row["val"][anchors]] = True
        row["calibrated_weights"], _ = calibrated_weights(
            pos,
            row["query"],
            row["labels"][row["query_local"]],
            np.flatnonzero(train_mask),
            official_counts,
            official_log_d1,
        )
        val, labels, _, action_logs, reconstruction_error = build_fold_action_logs(
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
        if not np.array_equal(val, row["val"]) or not np.array_equal(labels, row["labels"]):
            raise RuntimeError("Phase10 fold action logs do not match the fixed fold")
        row["phase10_raw_target"] = mixed_target(
            row["base"],
            labels,
            row["query_local"],
            action_logs,
            core_weights,
            joint_weights,
            COMPLEMENT_ALPHA,
        )
        row["phase8_target"] = normalize(
            0.80 * row["base"]
            + 0.20
            * local_prediction(
                pos[:, :2],
                shifts,
                target,
                np.flatnonzero(train_mask),
                row["query"],
                4,
                3.0,
                "horizontal",
            )
        ).astype(np.float32)
        row["phase5_pas"] = np.asarray(
            np.load(ROOT / f"matched_phase5_pas_band24_fold{fold}.npy", mmap_mode="r")[
                row["query_local"]
            ]
        )
        row["phase10_reconstruction_error"] = float(reconstruction_error)
        log("prepared_target_inputs", fold=fold, rows=len(row["query"]))

    phase9_gate_alpha: list[np.ndarray] = []
    phase10_gate_alpha: list[np.ndarray] = []
    for heldout in range(5):
        train_x = np.concatenate([folds[index]["x"] for index in range(5) if index != heldout])
        train_y = np.concatenate(
            [folds[index]["gain_grid"] for index in range(5) if index != heldout]
        )
        p9_weight = np.concatenate(
            [folds[index]["weights"] for index in range(5) if index != heldout]
        )
        p10_weight = np.concatenate(
            [folds[index]["calibrated_weights"] for index in range(5) if index != heldout]
        )
        for target_list, sample_weight in (
            (phase9_gate_alpha, p9_weight),
            (phase10_gate_alpha, p10_weight),
        ):
            model = ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=80,
                max_features=0.7,
                n_jobs=-1,
                random_state=52180,
            )
            model.fit(train_x, train_y, sample_weight=sample_weight)
            target_list.append(
                ALPHA_GRID[np.argmax(model.predict(folds[heldout]["x"]), axis=1)]
            )

    for fold, row in enumerate(folds):
        alpha9 = np.clip(P9_LOCAL_SCALE * phase9_gate_alpha[fold], 0.0, 0.6)[
            :, None, None, None
        ]
        corrected = normalize(row["base"] * np.exp(P9_RESIDUAL_ALPHA * row["pas_residual"]))
        row["phase9_pas"] = normalize(
            (1.0 - alpha9) * corrected + alpha9 * row["local"]
        ).astype(np.float32)
        row["phase9_pdp"] = normalize_last(
            row["base_pdp"] * np.exp(P9_PDP_ALPHA * row["pdp_residual"])
        ).astype(np.float32)
        labels = row["labels"][row["query_local"]]
        row["phase10_pas"] = apply_primary_anchor(
            row["phase10_raw_target"],
            row["pas_residual"],
            row["local"],
            labels,
            phase10_gate_alpha[fold],
        )
    return folds, channel, energy, official_counts, actual_counts


@torch.no_grad()
def run() -> None:
    predeclared = load_predeclared()
    if OUTPUT.exists():
        raise RuntimeError("Phase18 validation output already exists; first-run audit is required")
    folds, channel, energy, official_counts, actual_counts = prepare_all_targets()
    reference = selected_phase9_reference()
    standard_group_accumulators: list[np.ndarray] = []
    calibrated_group_accumulators: list[np.ndarray] = []
    fold_rows: list[dict[str, object]] = []
    all_finite_nonzero = True
    max_reference_error = 0.0

    for fold, row in enumerate(folds):
        candidate_pas, candidate_pdp, direction_diagnostic = secant_targets(
            row["base"],
            row["phase5_pas"],
            row["phase8_target"],
            row["phase9_pas"],
            row["phase10_pas"],
            row["base_pdp"],
            row["phase9_pdp"],
            row["weights"],
            ETA,
        )
        standard_accumulator = np.zeros((2, 6), dtype=np.float64)
        standard_group = np.zeros((2, 11, 6), dtype=np.float64)
        calibrated_accumulator = np.zeros((2, 6), dtype=np.float64)
        calibrated_group = np.zeros((2, 11, 6), dtype=np.float64)
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        labels = row["labels"][row["query_local"]]
        for start in range(0, len(row["query"]), 2):
            stop = min(start + 2, len(row["query"]))
            local_query = row["query_local"][start:stop]
            p = torch.as_tensor(np.asarray(prediction[local_query]).copy(), device=DEVICE)
            truth = torch.as_tensor(
                np.asarray(channel[row["query"][start:stop]]).copy(), device=DEVICE
            )
            truth_pas = torch.abs(rp.bs_fft_torch(truth)) ** 2
            truth_pdp = torch.abs(torch.fft.fft(truth, dim=-1, norm="ortho")) ** 2
            base_band = torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE)
            values = (
                project_joint(
                    p,
                    base_band,
                    torch.as_tensor(row["phase9_pas"][start:stop].copy(), device=DEVICE),
                    torch.as_tensor(row["phase9_pdp"][start:stop].copy(), device=DEVICE),
                    ITERATIONS,
                ),
                project_joint(
                    p,
                    base_band,
                    torch.as_tensor(candidate_pas[start:stop].copy(), device=DEVICE),
                    torch.as_tensor(candidate_pdp[start:stop].copy(), device=DEVICE),
                    ITERATIONS,
                ),
            )
            standard_weights = torch.as_tensor(
                row["weights"][start:stop].astype(np.float32), device=DEVICE
            )
            calibrated = torch.as_tensor(
                row["calibrated_weights"][start:stop].astype(np.float32), device=DEVICE
            )
            for index, value in enumerate(values):
                block = value.detach().cpu().numpy()
                all_finite_nonzero &= bool(
                    np.isfinite(block).all()
                    and np.all(np.sum(np.abs(block) ** 2, axis=(1, 2, 3)) > 0)
                )
                update_weighted_scores(
                    standard_accumulator,
                    standard_group,
                    index,
                    value,
                    truth,
                    truth_pas,
                    truth_pdp,
                    standard_weights,
                    labels[start:stop],
                )
                update_weighted_scores(
                    calibrated_accumulator,
                    calibrated_group,
                    index,
                    value,
                    truth,
                    truth_pas,
                    truth_pdp,
                    calibrated,
                    labels[start:stop],
                )
        standard = [components(standard_accumulator[index]) for index in range(2)]
        calibrated_values = [components(calibrated_accumulator[index]) for index in range(2)]
        error = max(
            abs(standard[0][name] - reference[fold][name])
            for name in ("c1_pas", "c2_pdp", "c3_nmse", "score")
        )
        max_reference_error = max(max_reference_error, error)
        fold_rows.append(
            {
                "fold": fold,
                "p9_reference": reference[fold],
                "p9_reconstructed_standard": standard[0],
                "candidate_standard": standard[1],
                "standard_delta_vs_p9": standard[1]["score"] - standard[0]["score"],
                "p9_calibrated": calibrated_values[0],
                "candidate_calibrated": calibrated_values[1],
                "calibrated_delta_vs_p9": (
                    calibrated_values[1]["score"] - calibrated_values[0]["score"]
                ),
                "p9_reference_max_component_error": error,
                "direction": direction_diagnostic,
            }
        )
        standard_group_accumulators.append(standard_group)
        calibrated_group_accumulators.append(calibrated_group)
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
    composites: dict[str, object] = {}
    for mapping_name, mapping in mappings.items():
        standard = [
            composite_components(standard_group_accumulators, index, mapping)
            for index in range(2)
        ]
        calibrated = [
            composite_components(calibrated_group_accumulators, index, mapping)
            for index in range(2)
        ]
        composites[mapping_name] = {
            "p9_standard": standard[0],
            "candidate_standard": standard[1],
            "standard_delta_vs_p9": standard[1]["score"] - standard[0]["score"],
            "p9_calibrated": calibrated[0],
            "candidate_calibrated": calibrated[1],
            "calibrated_delta_vs_p9": calibrated[1]["score"] - calibrated[0]["score"],
        }

    standard_delta = np.asarray(
        [float(row["standard_delta_vs_p9"]) for row in fold_rows], dtype=np.float64
    )
    calibrated_delta = np.asarray(
        [float(row["calibrated_delta_vs_p9"]) for row in fold_rows], dtype=np.float64
    )
    residual_fractions = np.asarray(
        [row["direction"]["bad_residual_norm_fraction"] for row in fold_rows],
        dtype=np.float64,
    )
    pas_clip = np.asarray(
        [row["direction"]["pas_projection_ratio_clip_fraction"] for row in fold_rows],
        dtype=np.float64,
    )
    rotation_delta = np.asarray(
        [
            composites[f"inner_{rotation}"]["calibrated_delta_vs_p9"]
            for rotation in range(1, 5)
        ],
        dtype=np.float64,
    )
    geometry_delta = float(np.dot(FOLD_WEIGHTS, calibrated_delta))
    locked_delta = float(composites["locked"]["calibrated_delta_vs_p9"])
    gates = predeclared["veto_gates"]
    checks = {
        "p9_reference_exact": max_reference_error
        <= float(gates["p9_reference_max_component_error"]),
        "bad_residual_fraction": bool(
            np.min(residual_fractions) >= float(gates["minimum_pas_bad_residual_fraction"])
        ),
        "pas_clip_fraction": bool(
            np.max(pas_clip) <= float(gates["maximum_eta1_pas_ratio_clip_fraction"])
        ),
        "fold_catastrophe": bool(
            np.min(calibrated_delta) >= float(gates["minimum_fold_delta_vs_p9"])
        ),
        "geometry_catastrophe": geometry_delta
        >= float(gates["minimum_geometry_weighted_delta_vs_p9"]),
        "locked_catastrophe": locked_delta >= float(gates["minimum_locked_delta_vs_p9"]),
        "rotation_catastrophe": bool(
            np.min(rotation_delta) >= float(gates["minimum_rotation_delta_vs_p9"])
        ),
        "finite_nonzero": all_finite_nonzero,
    }
    summary = {
        "standard_fold_deltas_vs_p9": standard_delta.tolist(),
        "calibrated_fold_deltas_vs_p9": calibrated_delta.tolist(),
        "standard_mean_delta_vs_p9": float(np.mean(standard_delta)),
        "calibrated_geometry_weighted_delta_vs_p9": geometry_delta,
        "locked_calibrated_delta_vs_p9": locked_delta,
        "rotation_calibrated_deltas_vs_p9": rotation_delta.tolist(),
        "minimum_bad_residual_norm_fraction": float(np.min(residual_fractions)),
        "maximum_pas_projection_ratio_clip_fraction": float(np.max(pas_clip)),
        "maximum_p9_reference_component_error": max_reference_error,
        "veto_checks": checks,
        "no_catastrophe_veto_passed": bool(all(checks.values())),
    }
    output = {
        "protocol": predeclared["protocol"],
        "predeclared_sha256": file_sha256(PREDECLARED),
        "validation_script_sha256_before_output": file_sha256(Path(__file__).resolve()),
        "eta": ETA,
        "truth_used_only_for_fold_scoring_after_target_freeze": True,
        "test_target_or_channel_built": False,
        "upload_performed": False,
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "zero_outliers": int(np.sum(energy == 0)),
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "locked_fold_by_group": LOCKED_FOLD_BY_GROUP,
        "folds": fold_rows,
        "composites": composites,
        "summary": summary,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log("validation_complete", output=OUTPUT.name, summary=summary)


if __name__ == "__main__":
    run()
