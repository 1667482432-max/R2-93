from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from build_phase9_submission import (
    gate_features_test,
    interpolate_pas_residual,
    official_anchors,
)
from matched_phase5_tree_band_descriptor import build_cache
from matched_phase5_tree_descriptor import model_features
from matched_phase6_pas_fitted_canonical import (
    direction,
    fit_coefficients as fit_h_coefficients,
    h_moment,
    shifts as h_shifts,
)
from matched_phase6_pas_fitted_vertical import (
    fit_coefficients as fit_v_coefficients,
    shifts as v_shifts,
    v_moment,
)
from matched_phase7_pas_aggregate_canonical import (
    aggregate,
    normalize_profile,
    roll_profile,
)
from matched_phase7_pas_aggregate_graph_metric import build_coordinates
from matched_phase7_pas_aggregate_local_gp import adaptive_gp
from matched_phase7_pas_aggregate_portfolio_exact import PORTFOLIOS
from matched_phase7_pas_harmonic_graph import harmonic_coefficients
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
from phase10_phase7_component_decompose import (
    action_weight_map,
    build_fold_action_logs,
    exact_target,
    load_selections,
)


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
ROBUST_ALPHA = 1.25
RESIDUAL_ALPHA = 0.10
LOCAL_SCALE = 0.50
ANCHOR_GROUPS = (4, 5, 10)
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
ROBUST_KEEP = {
    1: (0, 1, 2),
    3: (0, 1, 2),
    4: (0, 2),
    9: (0, 1),
    10: (0, 1, 2),
}
ROBUST_TEST_PATH = ROOT / "phase10_robust125_pas_band24_test.npy"
JOINT_TEST_PATH = ROOT / "phase10_robust125_primary_anchor_pas_band24_test.npy"
TEST_MANIFEST_PATH = ROOT / "phase10_robust125_primary_anchor_test_pas_manifest.json"
VALIDATION_PATH = ROOT / "phase10_robust125_primary_anchor_joint_validation.json"


def log(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}), flush=True)


def robust_weights() -> dict[tuple[int, int], float]:
    global_selection, _ = load_selections()
    weights = action_weight_map(global_selection)
    for group, actions in PORTFOLIOS.items():
        keep = set(ROBUST_KEEP.get(group, ()))
        for index in range(len(actions)):
            if index not in keep:
                weights[(group, index)] = 0.0
    return weights


def weights_manifest(weights: dict[tuple[int, int], float]) -> dict[str, list[dict]]:
    return {
        str(group): [
            {
                "index": index,
                "family": action[0],
                "parameters": list(action[1]),
                "action_exponent": float(action[2]),
                "portfolio_weight": float(weights[(group, index)]),
                "effective_log_exponent": float(
                    action[2] * weights[(group, index)]
                ),
            }
            for index, action in enumerate(actions)
            if weights[(group, index)] != 0.0
        ]
        for group, actions in PORTFOLIOS.items()
    }


def composite_components(
    group_accumulators: list[np.ndarray], index: int, mapping: dict[int, int]
) -> dict[str, float]:
    total = np.zeros(6, dtype=np.float64)
    for group, fold in mapping.items():
        total += group_accumulators[fold][index, group]
    return components(total)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize_profile_bs(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def build_robust_test_source(
    pos: np.ndarray,
    test_pos: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    weights: dict[tuple[int, int], float],
    group_alphas: dict[int, float] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    all_pos = np.vstack((pos, test_pos))
    test_index = np.arange(len(pos), len(all_pos))
    aggregate_target = aggregate(target)
    base = np.asarray(
        np.load(
            ROOT / "matched_phase6_milestone_physics_pas_band24_test.npy",
            mmap_mode="r",
        )
    )
    base_profile = aggregate(base)
    coordinates = build_coordinates(all_pos, valid)
    unit, side = direction(all_pos)
    horizontal_moment = h_moment(target, valid)
    vertical_moment = v_moment(target, valid)
    h_coefficient = fit_h_coefficients(
        unit, side, valid, valid, horizontal_moment, 60050
    )
    v_coefficient = fit_v_coefficients(
        unit, side, valid, valid, vertical_moment, 60150
    )
    horizontal_shift = h_shifts(unit, side, h_coefficient)
    vertical_shift = v_shifts(unit, side, v_coefficient)
    anchor_h_coefficient = fit_h_coefficients(
        unit, side, valid, valid, horizontal_moment, 59850
    )
    anchor_v_coefficient = fit_v_coefficients(
        unit, side, valid, valid, vertical_moment, 59950
    )
    anchor_horizontal_shift = h_shifts(unit, side, anchor_h_coefficient)
    anchor_vertical_shift = v_shifts(unit, side, anchor_v_coefficient)
    rich_features = model_features(
        all_pos, np.load(ROOT / "rich_map_features.npy").astype(np.float32)
    )
    candidate_cache: dict[tuple, np.ndarray] = {}
    pca_cache: dict[tuple[str, str], tuple[PCA, np.ndarray]] = {}
    canonical_cache: dict[str, np.ndarray] = {}
    anchor_counts: dict[str, int] = {}

    def canonical_training(family: str, variant: str) -> tuple[PCA, np.ndarray]:
        key = (family, variant)
        if key in pca_cache:
            return pca_cache[key]
        h_train = (
            -horizontal_shift[valid]
            if variant in ("h", "hv")
            else np.zeros(len(valid), int)
        )
        v_train = (
            -vertical_shift[valid]
            if variant == "hv"
            else np.zeros(len(valid), int)
        )
        canonical = roll_profile(aggregate_target[valid], h_train, v_train)
        seed_base = {"harmonic": 58800, "graph": 59200, "gp": 59700}[family]
        pca = PCA(
            n_components=64,
            svd_solver="randomized",
            random_state=seed_base + 50 + ("plain", "h", "hv").index(variant),
        )
        coefficient = pca.fit_transform(
            np.sqrt(np.maximum(canonical, 0))
        ).astype(np.float32)
        pca_cache[key] = (pca, coefficient)
        return pca, coefficient

    def restore_orientation(value: np.ndarray, variant: str) -> np.ndarray:
        h_test = (
            horizontal_shift[test_index]
            if variant in ("h", "hv")
            else np.zeros(len(test_pos), int)
        )
        v_test = (
            vertical_shift[test_index]
            if variant == "hv"
            else np.zeros(len(test_pos), int)
        )
        return normalize_profile(
            np.maximum(roll_profile(value, h_test, v_test), 0) ** 2
        ).astype(np.float32)

    def canonical_candidate(name: str) -> np.ndarray:
        if name in canonical_cache:
            return canonical_cache[name]
        variant, model_name = name.split("_", 1)
        h_train = (
            -horizontal_shift[valid]
            if variant in ("h", "hv")
            else np.zeros(len(valid), int)
        )
        v_train = (
            -vertical_shift[valid]
            if variant == "hv"
            else np.zeros(len(valid), int)
        )
        canonical = roll_profile(aggregate_target[valid], h_train, v_train)
        pca = PCA(
            n_components=64,
            svd_solver="randomized",
            random_state=58350 + ("plain", "h", "hv").index(variant),
        )
        coefficient = pca.fit_transform(
            np.sqrt(np.maximum(canonical, 0))
        ).astype(np.float32)
        if model_name == "rbf":
            spatial_scale = np.std(pos[valid, :2], axis=0)
            model = RBFInterpolator(
                pos[valid, :2] / spatial_scale,
                coefficient,
                neighbors=100,
                smoothing=0.1,
                kernel="linear",
                degree=0,
            )
            predicted = model(test_pos[:, :2] / spatial_scale)
        elif model_name.startswith("extra"):
            leaf = int(model_name.removeprefix("extra"))
            model = ExtraTreesRegressor(
                n_estimators=600,
                min_samples_leaf=leaf,
                max_features=0.65,
                n_jobs=-1,
                random_state=58900 + leaf,
            )
            model.fit(rich_features[valid], coefficient)
            predicted = model.predict(rich_features[test_index])
        elif model_name == "forest3":
            model = RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                max_features=0.65,
                n_jobs=-1,
                random_state=59000,
            )
            model.fit(rich_features[valid], coefficient)
            predicted = model.predict(rich_features[test_index])
        else:
            raise ValueError(name)
        output = restore_orientation(pca.inverse_transform(predicted), variant)
        canonical_cache[name] = output
        return output

    def candidate(family: str, parameters: tuple, group: int) -> np.ndarray:
        key = (family, parameters, group if family == "anchor" else -1)
        if key in candidate_cache:
            return candidate_cache[key]
        if family == "canonical":
            output = canonical_candidate(parameters[0])
        elif family in ("harmonic", "graph"):
            if family == "harmonic":
                variant, neighbors, kernel = parameters
                node_coordinates = all_pos[
                    np.concatenate((valid, test_index)), :2
                ].astype(np.float32)
            else:
                variant, mode, neighbors, kernel = parameters
                node_coordinates = coordinates[mode][
                    np.concatenate((valid, test_index))
                ]
            pca, coefficient = canonical_training(family, variant)
            predicted = harmonic_coefficients(
                node_coordinates, len(valid), coefficient, neighbors, kernel
            )
            output = restore_orientation(pca.inverse_transform(predicted), variant)
        elif family == "gp":
            variant, k, kernel, factor, nugget, mode = parameters
            pca, coefficient = canonical_training(family, variant)
            predicted = adaptive_gp(
                pos[valid, :2].astype(np.float32),
                test_pos[:, :2].astype(np.float32),
                coefficient,
                k,
                kernel,
                factor,
                nugget,
                mode,
            )
            output = restore_orientation(pca.inverse_transform(predicted), variant)
        elif family == "anchor":
            variant, method = parameters
            output = base_profile.copy()
            mask = labels == group
            query = test_pos[mask, :2]
            lo, hi = query.min(0), query.max(0)
            inside = np.all(pos[:, :2] >= lo, axis=1) & np.all(
                pos[:, :2] <= hi, axis=1
            )
            anchors = np.flatnonzero(inside & np.isin(np.arange(len(pos)), valid))
            anchor_counts[f"g{group}_{variant}_{method}"] = int(len(anchors))
            if len(anchors):
                distance, local = cKDTree(pos[anchors, :2]).query(
                    query, k=len(anchors)
                )
                if len(anchors) == 1:
                    distance, local = distance[:, None], local[:, None]
                h_anchor = (
                    -anchor_horizontal_shift[anchors]
                    if variant in ("h", "hv")
                    else np.zeros(len(anchors), int)
                )
                v_anchor = (
                    -anchor_vertical_shift[anchors]
                    if variant == "hv"
                    else np.zeros(len(anchors), int)
                )
                source = roll_profile(aggregate_target[anchors], h_anchor, v_anchor)
                if method == "nearest":
                    value = source[local[:, 0]]
                elif method == "mean":
                    value = np.broadcast_to(source.mean(0), (mask.sum(), 256))
                else:
                    power = int(method.removeprefix("idw"))
                    local_weight = 1.0 / np.maximum(distance, 0.5) ** power
                    local_weight /= local_weight.sum(1, keepdims=True)
                    value = np.einsum(
                        "nk,nkp->np", local_weight, source[local], optimize=True
                    )
                query_indices = test_index[mask]
                h_query = (
                    anchor_horizontal_shift[query_indices]
                    if variant in ("h", "hv")
                    else np.zeros(mask.sum(), int)
                )
                v_query = (
                    anchor_vertical_shift[query_indices]
                    if variant == "hv"
                    else np.zeros(mask.sum(), int)
                )
                output[mask] = normalize_profile(
                    np.maximum(roll_profile(value, h_query, v_query), 0)
                )
        else:
            raise ValueError(family)
        candidate_cache[key] = output
        return output

    source = base.copy()
    epsilon = 1e-3 / base.shape[1]
    active_groups = sorted(
        group
        for group, actions in PORTFOLIOS.items()
        if any(weights[(group, index)] != 0.0 for index in range(len(actions)))
    )
    for group in active_groups:
        actions = PORTFOLIOS[group]
        mask = labels == group
        log_ratio = np.zeros((mask.sum(), 256), dtype=np.float64)
        for index, (family, parameters, exponent) in enumerate(actions):
            weight = weights[(group, index)]
            if weight == 0.0:
                continue
            profile = candidate(family, tuple(parameters), group)[mask]
            ratio = np.clip(
                (profile + epsilon) / (base_profile[mask] + epsilon), 0.25, 4.0
            )
            log_ratio += weight * float(exponent) * np.log(ratio)
        source[mask] = normalize(
            base[mask] * np.exp(log_ratio)[:, :, None, None]
        )
        log("test_robust_group", group=group, rows=int(mask.sum()))

    robust = base.copy()
    alpha_by_group = {
        group: float(
            ROBUST_ALPHA if group_alphas is None else group_alphas[group]
        )
        for group in active_groups
    }
    for group in active_groups:
        mask = labels == group
        alpha = alpha_by_group[group]
        robust[mask] = normalize(
            (1.0 - alpha) * base[mask] + alpha * source[mask]
        )
    robust = robust.astype(np.float32)
    diagnostics = {
        "active_groups": active_groups,
        "alpha_by_group": {str(k): v for k, v in alpha_by_group.items()},
        "candidate_count": len(candidate_cache),
        "canonical_count": len(canonical_cache),
        "anchor_counts_by_action": anchor_counts,
        "robust_min": float(robust.min()),
        "robust_negative_fraction": float(np.mean(robust < 0)),
    }
    return robust, diagnostics


def build_test_targets(
    folds: list[dict],
    pos: np.ndarray,
    energy: np.ndarray,
    official_counts: dict[int, int],
    weights: dict[tuple[int, int], float],
    validation_summary: dict[str, object],
) -> dict[str, object]:
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    valid = np.flatnonzero(energy > 0)
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    robust, robust_diagnostic = build_robust_test_source(
        pos, test_pos, valid, target, test_labels, weights
    )
    np.save(ROBUST_TEST_PATH, robust)
    log("test_robust_saved", path=ROBUST_TEST_PATH.name)

    anchors, anchor_labels, _, anchor_counts = official_anchors(pos, energy, test_pos)
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
        all_pos[:, :2], shifts, target, external_train, anchors, 4, 3.0, "none"
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
    group_mask = np.isin(test_labels, ANCHOR_GROUPS).astype(np.float32)
    alpha = (
        group_mask * np.clip(LOCAL_SCALE * raw_gate, 0.0, 0.30)
    )[:, None, None, None]
    corrected = normalize(
        robust * np.exp(RESIDUAL_ALPHA * group_mask[:, None, None, None] * residual)
    )
    joint = normalize((1.0 - alpha) * corrected + alpha * local).astype(np.float32)
    np.save(JOINT_TEST_PATH, joint)
    log("test_joint_saved", path=JOINT_TEST_PATH.name)

    manifest = {
        "validation_passed": True,
        "validation_criterion": "locked edge delta >= 0.005 and all five fold deltas > 0",
        "validation_summary": validation_summary,
        "base": "matched_phase6_milestone_physics_pas_band24_test.npy",
        "robust_pas_target": {
            "path": ROBUST_TEST_PATH.name,
            "shape": list(robust.shape),
            "dtype": str(robust.dtype),
            "bytes": ROBUST_TEST_PATH.stat().st_size,
            "sha256": sha256(ROBUST_TEST_PATH),
        },
        "joint_pas_target": {
            "path": JOINT_TEST_PATH.name,
            "shape": list(joint.shape),
            "dtype": str(joint.dtype),
            "bytes": JOINT_TEST_PATH.stat().st_size,
            "sha256": sha256(JOINT_TEST_PATH),
            "min": float(joint.min()),
            "negative_fraction": float(np.mean(joint < 0)),
        },
        "parameters": {
            "robust_alpha": ROBUST_ALPHA,
            "residual_alpha": RESIDUAL_ALPHA,
            "local_scale": LOCAL_SCALE,
            "local_clip": 0.30,
            "anchor_groups": ANCHOR_GROUPS,
            "projection_bands": 24,
            "projection_iterations": ITERATIONS,
            "gate": {
                "model": "ExtraTreesRegressor",
                "n_estimators": 500,
                "min_samples_leaf": 80,
                "max_features": 0.7,
                "random_state": 52180,
                "training_weight": "difficulty-calibrated official-count weights",
            },
        },
        "test_geometry": {
            "official_counts": {str(k): int(v) for k, v in official_counts.items()},
            "anchor_count": int(len(anchors)),
            "anchor_counts": {str(k): int(v) for k, v in anchor_counts.items()},
            "gate_raw_quantiles": np.quantile(
                raw_gate, [0.0, 0.1, 0.5, 0.9, 1.0]
            ).tolist(),
            "applied_alpha_mean_all": float(alpha.mean()),
            "applied_alpha_mean_anchor_groups": float(
                alpha[group_mask.astype(bool)].mean()
            ),
        },
        "robust_diagnostic": robust_diagnostic,
        "robust_actions": weights_manifest(weights),
        "next_step": "project Phase6 channel toward joint_pas_target with 24 bands and 4 iterations; no PDP target change",
    }
    TEST_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("test_manifest_saved", path=TEST_MANIFEST_PATH.name)
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
    weights = robust_weights()

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
        row["robust_target"] = exact_target(
            old_base,
            row["base"],
            labels,
            row["query_local"],
            action_logs,
            weights,
            legacy_old_base=False,
            pas_alpha=ROBUST_ALPHA,
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
        log(
            "prepare_fold",
            fold=fold,
            reconstruction_error=reconstruction_error,
        )

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

    fold_rows = []
    group_accumulators: list[np.ndarray] = []
    for fold, row in enumerate(folds):
        labels = row["labels"][row["query_local"]]
        group_mask = np.isin(labels, ANCHOR_GROUPS).astype(np.float32)[
            :, None, None, None
        ]
        corrected = normalize(
            row["robust_target"]
            * np.exp(RESIDUAL_ALPHA * group_mask * row["residual"])
        )
        alpha = group_mask * np.clip(
            LOCAL_SCALE * gate_alpha[fold], 0.0, 0.30
        )[:, None, None, None]
        joint_target = normalize(
            (1.0 - alpha) * corrected + alpha * row["local"]
        ).astype(np.float32)
        prediction = np.load(
            ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r"
        )
        accumulator = np.zeros((3, 6), dtype=np.float64)
        group_accumulator = np.zeros((3, 11, 6), dtype=np.float64)
        batch_size = 8
        for start in range(0, len(row["query"]), batch_size):
            stop = min(start + batch_size, len(row["query"]))
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
            base_band = torch.as_tensor(
                row["base"][start:stop].copy(), device=DEVICE
            )
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
            robust_value = project(
                p,
                base_band,
                torch.as_tensor(
                    row["robust_target"][start:stop].copy(), device=DEVICE
                ),
                24,
                ITERATIONS,
            )
            update_weighted_scores(
                accumulator,
                group_accumulator,
                1,
                robust_value,
                t,
                truth_pas,
                truth_pdp,
                batch_weights,
                labels[start:stop],
            )
            joint_value = project(
                p,
                base_band,
                torch.as_tensor(joint_target[start:stop].copy(), device=DEVICE),
                24,
                ITERATIONS,
            )
            update_weighted_scores(
                accumulator,
                group_accumulator,
                2,
                joint_value,
                t,
                truth_pas,
                truth_pdp,
                batch_weights,
                labels[start:stop],
            )
        phase6 = components(accumulator[0])
        robust = components(accumulator[1])
        joint = components(accumulator[2])
        fold_rows.append(
            {
                "fold": fold,
                "phase6": phase6,
                "robust": {
                    **robust,
                    "delta_vs_phase6": robust["score"] - phase6["score"],
                },
                "joint": {
                    **joint,
                    "delta_vs_phase6": joint["score"] - phase6["score"],
                    "increment_vs_robust": joint["score"] - robust["score"],
                },
                "gate_raw_quantiles": np.quantile(
                    gate_alpha[fold], [0.0, 0.1, 0.5, 0.9, 1.0]
                ).tolist(),
                "applied_alpha_mean": float(alpha.mean()),
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
    for name, mapping in mappings.items():
        phase6 = composite_components(group_accumulators, 0, mapping)
        robust = composite_components(group_accumulators, 1, mapping)
        joint = composite_components(group_accumulators, 2, mapping)
        composites[name] = {
            "phase6": phase6,
            "robust": {
                **robust,
                "delta_vs_phase6": robust["score"] - phase6["score"],
            },
            "joint": {
                **joint,
                "delta_vs_phase6": joint["score"] - phase6["score"],
                "increment_vs_robust": joint["score"] - robust["score"],
            },
        }

    robust_fold_delta = np.asarray(
        [row["robust"]["delta_vs_phase6"] for row in fold_rows]
    )
    joint_fold_delta = np.asarray(
        [row["joint"]["delta_vs_phase6"] for row in fold_rows]
    )
    anchor_increment = np.asarray(
        [row["joint"]["increment_vs_robust"] for row in fold_rows]
    )
    inner_anchor_increments = np.asarray(
        [
            composites[f"inner_{rotation}"]["joint"]["increment_vs_robust"]
            for rotation in range(1, 5)
        ]
    )
    summary = {
        "robust_fold_deltas_vs_phase6": robust_fold_delta.tolist(),
        "joint_fold_deltas_vs_phase6": joint_fold_delta.tolist(),
        "fold_anchor_increments_vs_robust": anchor_increment.tolist(),
        "robust_geometry_weighted_delta": float(
            np.dot(FOLD_WEIGHTS, robust_fold_delta)
        ),
        "joint_geometry_weighted_delta": float(
            np.dot(FOLD_WEIGHTS, joint_fold_delta)
        ),
        "joint_min_fold_delta": float(joint_fold_delta.min()),
        "joint_lcb_delta": float(
            joint_fold_delta.mean() - 0.75 * joint_fold_delta.std()
        ),
        "locked_edge_robust_delta": composites["locked"]["robust"][
            "delta_vs_phase6"
        ],
        "locked_edge_joint_delta": composites["locked"]["joint"][
            "delta_vs_phase6"
        ],
        "locked_edge_anchor_increment": composites["locked"]["joint"][
            "increment_vs_robust"
        ],
        "inner_anchor_increments": inner_anchor_increments.tolist(),
        "inner_anchor_positive_count": int(np.sum(inner_anchor_increments > 0)),
        "inner_anchor_mean": float(inner_anchor_increments.mean()),
        "all_folds_positive": bool(np.all(joint_fold_delta > 0)),
    }
    passed = bool(
        summary["locked_edge_joint_delta"] >= 0.005
        and summary["all_folds_positive"]
    )
    summary["test_target_build_criterion_passed"] = passed
    output = {
        "frozen_before_evaluation": True,
        "parameters": {
            "robust_alpha": ROBUST_ALPHA,
            "residual_alpha": RESIDUAL_ALPHA,
            "local_scale": LOCAL_SCALE,
            "local_clip": 0.30,
            "anchor_groups": ANCHOR_GROUPS,
            "projection_bands": 24,
            "projection_iterations": ITERATIONS,
            "no_op_semantics": "action ratios transferred onto anchor-retained Phase6 base; inactive groups equal base exactly",
        },
        "robust_actions": weights_manifest(weights),
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "locked_fold_by_group": LOCKED_FOLD_BY_GROUP,
        "official_log_d1": official_log_d1,
        "reconstruction_max_errors": reconstruction_errors,
        "calibration_diagnostics": calibration_diagnostics,
        "folds": fold_rows,
        "composites": composites,
        "summary": summary,
    }
    VALIDATION_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log("validation_saved", path=VALIDATION_PATH.name, summary=summary)

    if passed:
        test_manifest = build_test_targets(
            folds, pos, energy, official_counts, weights, summary
        )
        output["test_target_manifest"] = test_manifest
        VALIDATION_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    else:
        log("test_target_skipped", reason="validation criterion not met")


if __name__ == "__main__":
    run()
