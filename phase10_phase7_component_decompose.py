from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import build_cache
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
    point_cosine,
    roll_profile,
)
from matched_phase7_pas_aggregate_graph_metric import build_coordinates
from matched_phase7_pas_aggregate_local_gp import adaptive_gp
from matched_phase7_pas_aggregate_portfolio_exact import PORTFOLIOS
from matched_phase7_pas_harmonic_graph import harmonic_coefficients
from phase8_anchor_local_gate_channel_validation import components, prepare_folds
from phase8_anchor_retained_pas_resolution_validation import project
from phase8_anchor_retained_pas_screen import normalize
from phase8_anchor_retained_pdp_screen import official_geometry
from phase10_calibrated_pas_residual_joint_validation import (
    COMPOSITE_FOLD_BY_GROUP,
    FOLD_WEIGHTS,
    calibrated_weights,
    update_weighted_scores,
)


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
PAS_ALPHA = 0.75
PROJECTION_ITERATIONS = 4
EPSILON = 1e-3 / 256


def load_selections() -> tuple[dict, dict]:
    global_payload = json.loads(
        (ROOT / "matched_phase7_pas_aggregate_global_selection.json").read_text()
    )
    global_selection = next(
        row
        for row in global_payload["results"]
        if row["mode"] == "lcb" and float(row["penalty"]) == 0.0
    )["selected"]
    exact_payload = json.loads(
        (ROOT / "matched_phase7_pas_aggregate_portfolio_exact.json").read_text()
    )
    return global_selection, exact_payload["selected"]


def family_bucket(family: str, parameters: tuple) -> str:
    if family != "canonical":
        return family
    return "canonical_rbf" if "rbf" in str(parameters[0]) else "canonical_tree"


def normalize_2d(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(value, weights=weight))


def profile_cosine(value: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sum(value * truth, axis=1) / np.maximum(
        np.linalg.norm(value, axis=1) * np.linalg.norm(truth, axis=1), 1e-30
    )


def action_weight_map(selection: dict) -> dict[tuple[int, int], float]:
    return {
        (group, index): float(weight)
        for group, actions in PORTFOLIOS.items()
        for index, weight in enumerate(selection[str(group)]["action_weights"])
    }


def config_weights(
    global_weights: dict[tuple[int, int], float],
    predicate,
) -> dict[tuple[int, int], float]:
    return {
        key: value if predicate(key[0], key[1], PORTFOLIOS[key[0]][key[1]]) else 0.0
        for key, value in global_weights.items()
    }


def build_fold_action_logs(
    fold: int,
    pos: np.ndarray,
    energy: np.ndarray,
    target: np.ndarray,
    aggregate_target: np.ndarray,
    unit: np.ndarray,
    side: np.ndarray,
    horizontal_moment: np.ndarray,
    vertical_moment: np.ndarray,
    coordinates: dict[str, np.ndarray],
    diagnostics: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray], float]:
    valid_mask = energy > 0
    valid = np.flatnonzero(valid_mask)
    val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
    labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
    train = valid[~np.isin(valid, val)]

    h_coefficient = fit_h_coefficients(
        unit, side, valid, train, horizontal_moment, 60000 + 10 * fold
    )
    v_coefficient = fit_v_coefficients(
        unit, side, valid, train, vertical_moment, 60100 + 10 * fold
    )
    horizontal_shift = h_shifts(unit, side, h_coefficient)
    vertical_shift = v_shifts(unit, side, v_coefficient)
    anchor_h_coefficient = fit_h_coefficients(
        unit, side, valid, train, horizontal_moment, 59800 + 10 * fold
    )
    anchor_v_coefficient = fit_v_coefficients(
        unit, side, valid, train, vertical_moment, 59900 + 10 * fold
    )
    anchor_horizontal_shift = h_shifts(unit, side, anchor_h_coefficient)
    anchor_vertical_shift = v_shifts(unit, side, anchor_v_coefficient)

    old_base = np.asarray(
        np.load(
            ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
            mmap_mode="r",
        )
    )
    old_base_profile = aggregate(old_base)
    cache: dict[tuple, np.ndarray] = {}
    pca_cache: dict[tuple[str, str], tuple[PCA, np.ndarray]] = {}

    def canonical_training(family: str, variant: str) -> tuple[PCA, np.ndarray]:
        key = (family, variant)
        if key in pca_cache:
            return pca_cache[key]
        h_train = (
            -horizontal_shift[train]
            if variant in ("h", "hv")
            else np.zeros(len(train), int)
        )
        v_train = (
            -vertical_shift[train]
            if variant == "hv"
            else np.zeros(len(train), int)
        )
        canonical = roll_profile(aggregate_target[train], h_train, v_train)
        seed_base = {"harmonic": 58800, "graph": 59200, "gp": 59700}[family]
        pca = PCA(
            n_components=64,
            svd_solver="randomized",
            random_state=seed_base
            + 10 * fold
            + ("plain", "h", "hv").index(variant),
        )
        coefficient = pca.fit_transform(
            np.sqrt(np.maximum(canonical, 0))
        ).astype(np.float32)
        pca_cache[key] = (pca, coefficient)
        return pca, coefficient

    def restore_orientation(value: np.ndarray, variant: str) -> np.ndarray:
        h_val = (
            horizontal_shift[val]
            if variant in ("h", "hv")
            else np.zeros(len(val), int)
        )
        v_val = vertical_shift[val] if variant == "hv" else np.zeros(len(val), int)
        return normalize_profile(
            np.maximum(roll_profile(value, h_val, v_val), 0) ** 2
        ).astype(np.float32)

    def candidate(family: str, parameters: tuple, group: int) -> np.ndarray:
        key = (family, parameters, group if family == "anchor" else -1)
        if key in cache:
            return cache[key]
        if family == "canonical":
            name = parameters[0]
            output = np.asarray(
                np.load(
                    ROOT / f"matched_phase7_aggregate_{name}_fold{fold}.npy",
                    mmap_mode="r",
                )
            )
        elif family in ("harmonic", "graph"):
            if family == "harmonic":
                variant, neighbors, kernel = parameters
                node_coordinates = pos[np.concatenate((train, val)), :2].astype(
                    np.float32
                )
            else:
                variant, mode, neighbors, kernel = parameters
                node_coordinates = coordinates[mode][np.concatenate((train, val))]
            pca, coefficient = canonical_training(family, variant)
            predicted = harmonic_coefficients(
                node_coordinates, len(train), coefficient, neighbors, kernel
            )
            output = restore_orientation(pca.inverse_transform(predicted), variant)
        elif family == "gp":
            variant, k, kernel, factor, nugget, mode = parameters
            pca, coefficient = canonical_training(family, variant)
            predicted = adaptive_gp(
                pos[train, :2].astype(np.float32),
                pos[val, :2].astype(np.float32),
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
            output = old_base_profile.copy()
            item = next(
                row
                for row in diagnostics
                if row["fold"] == fold and row["group"] == group
            )
            lo, hi = np.asarray(item["box_lo"]), np.asarray(item["box_hi"])
            inside = (
                valid_mask
                & np.all(pos[:, :2] >= lo, axis=1)
                & np.all(pos[:, :2] <= hi, axis=1)
            )
            anchors = np.flatnonzero(inside & ~np.isin(np.arange(len(pos)), val))
            mask = labels == group
            query = val[mask]
            if len(anchors):
                distance, local = cKDTree(pos[anchors, :2]).query(
                    pos[query, :2], k=len(anchors)
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
                    value = np.broadcast_to(source.mean(0), (len(query), 256))
                else:
                    power = int(method.removeprefix("idw"))
                    local_weight = 1.0 / np.maximum(distance, 0.5) ** power
                    local_weight /= local_weight.sum(1, keepdims=True)
                    value = np.einsum(
                        "nk,nkp->np", local_weight, source[local], optimize=True
                    )
                h_query = (
                    anchor_horizontal_shift[query]
                    if variant in ("h", "hv")
                    else np.zeros(len(query), int)
                )
                v_query = (
                    anchor_vertical_shift[query]
                    if variant == "hv"
                    else np.zeros(len(query), int)
                )
                output[mask] = normalize_profile(
                    np.maximum(roll_profile(value, h_query, v_query), 0)
                )
        else:
            raise ValueError(family)
        cache[key] = output
        return output

    logs: dict[tuple[int, int], np.ndarray] = {}
    for group, actions in PORTFOLIOS.items():
        mask = labels == group
        current = old_base_profile[mask]
        for index, (family, parameters, exponent) in enumerate(actions):
            desired = candidate(family, tuple(parameters), group)[mask]
            ratio = np.clip((desired + EPSILON) / (current + EPSILON), 0.25, 4.0)
            logs[(group, index)] = (float(exponent) * np.log(ratio)).astype(np.float32)

    global_selection, _ = load_selections()
    global_weights = action_weight_map(global_selection)
    reconstructed = old_base.copy()
    for group, actions in PORTFOLIOS.items():
        mask = labels == group
        total = sum(
            global_weights[(group, index)] * logs[(group, index)]
            for index in range(len(actions))
        )
        reconstructed[mask] = normalize(
            old_base[mask] * np.exp(total)[:, :, None, None]
        )
    saved = np.load(
        ROOT / f"matched_phase7_aggregate_portfolio_mean_pas_band24_fold{fold}.npy",
        mmap_mode="r",
    )
    max_reconstruction_error = float(
        np.max(np.abs(reconstructed.astype(np.float32) - np.asarray(saved)))
    )
    return val, labels, old_base, logs, max_reconstruction_error


def approximate_target(
    old_base_profile: np.ndarray,
    new_base_profile: np.ndarray,
    labels: np.ndarray,
    query_local: np.ndarray,
    logs: dict[tuple[int, int], np.ndarray],
    weights: dict[tuple[int, int], float],
    legacy_old_base: bool = False,
) -> np.ndarray:
    source_base = (
        old_base_profile[query_local] if legacy_old_base else new_base_profile
    )
    output = source_base.copy()
    query_labels = labels[query_local]
    for group, actions in PORTFOLIOS.items():
        full_rows = np.flatnonzero(labels == group)
        query_rows = np.flatnonzero(query_labels == group)
        if not len(query_rows):
            continue
        keep = np.isin(full_rows, query_local)
        total = sum(
            weights[(group, index)] * logs[(group, index)][keep]
            for index in range(len(actions))
        )
        output[query_rows] = normalize_2d(
            source_base[query_rows] * np.exp(total)
        )
    return normalize_2d((1.0 - PAS_ALPHA) * new_base_profile + PAS_ALPHA * output)


def exact_target(
    old_base: np.ndarray,
    new_base: np.ndarray,
    labels: np.ndarray,
    query_local: np.ndarray,
    logs: dict[tuple[int, int], np.ndarray],
    weights: dict[tuple[int, int], float],
    legacy_old_base: bool = False,
    pas_alpha: float = PAS_ALPHA,
) -> np.ndarray:
    source_base = old_base[query_local] if legacy_old_base else new_base
    source = source_base.copy()
    query_labels = labels[query_local]
    for group, actions in PORTFOLIOS.items():
        full_rows = np.flatnonzero(labels == group)
        query_rows = np.flatnonzero(query_labels == group)
        if not len(query_rows):
            continue
        keep = np.isin(full_rows, query_local)
        total = sum(
            weights[(group, index)] * logs[(group, index)][keep]
            for index in range(len(actions))
        )
        source[query_rows] = normalize(
            source_base[query_rows] * np.exp(total)[:, :, None, None]
        )
    return normalize((1.0 - pas_alpha) * new_base + pas_alpha * source).astype(
        np.float32
    )


@torch.no_grad()
def run() -> None:
    folds, pos, channel, energy, official_counts, actual_counts = prepare_folds()
    target = build_cache()
    aggregate_target = aggregate(target)
    valid = energy > 0
    unit, side = direction(pos)
    valid_index = np.flatnonzero(valid)
    horizontal_moment = h_moment(target, valid_index)
    vertical_moment = v_moment(target, valid_index)
    coordinates = build_coordinates(pos, valid_index)
    diagnostics = json.loads((ROOT / "matched_rect_split_diagnostics.json").read_text())
    global_selection, safe_selection = load_selections()
    global_weights = action_weight_map(global_selection)
    safe_weights = action_weight_map(safe_selection)

    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    official_distance = np.asarray(cKDTree(pos[valid, :2]).query(test_pos[:, :2], k=1)[0])
    official_log_d1 = {
        int(group): float(np.mean(np.log1p(official_distance[test_labels == group])))
        for group in np.unique(test_labels)
    }
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)

    fixed_configs: dict[str, dict[tuple[int, int], float]] = {
        "full_global_legacy": global_weights,
        "full_global": global_weights,
        "full_safe_weights": safe_weights,
        "density_graph_harmonic": config_weights(
            global_weights, lambda g, i, a: a[0] in ("graph", "harmonic")
        ),
        "non_density_all": config_weights(
            global_weights, lambda g, i, a: a[0] not in ("graph", "harmonic")
        ),
        "canonical_anchor": config_weights(
            global_weights, lambda g, i, a: a[0] in ("canonical", "anchor")
        ),
        "canonical_gp": config_weights(
            global_weights, lambda g, i, a: a[0] in ("canonical", "gp")
        ),
        "rbf_gp": config_weights(
            global_weights,
            lambda g, i, a: family_bucket(a[0], tuple(a[1]))
            in ("canonical_rbf", "gp"),
        ),
        "tree_gp": config_weights(
            global_weights,
            lambda g, i, a: family_bucket(a[0], tuple(a[1]))
            in ("canonical_tree", "gp"),
        ),
        "graph_canonical": config_weights(
            global_weights, lambda g, i, a: a[0] in ("graph", "canonical")
        ),
        "graph_canonical_gp": config_weights(
            global_weights,
            lambda g, i, a: a[0] in ("graph", "canonical", "gp"),
        ),
        "canonical_only": config_weights(
            global_weights, lambda g, i, a: a[0] == "canonical"
        ),
        "canonical_tree_only": config_weights(
            global_weights,
            lambda g, i, a: family_bucket(a[0], tuple(a[1])) == "canonical_tree",
        ),
        "canonical_rbf_only": config_weights(
            global_weights,
            lambda g, i, a: family_bucket(a[0], tuple(a[1])) == "canonical_rbf",
        ),
        "tree_anchor": config_weights(
            global_weights,
            lambda g, i, a: family_bucket(a[0], tuple(a[1]))
            in ("canonical_tree", "anchor"),
        ),
        "rbf_anchor": config_weights(
            global_weights,
            lambda g, i, a: family_bucket(a[0], tuple(a[1]))
            in ("canonical_rbf", "anchor"),
        ),
        "anchor_only": config_weights(
            global_weights, lambda g, i, a: a[0] == "anchor"
        ),
        "gp_only": config_weights(global_weights, lambda g, i, a: a[0] == "gp"),
        "graph_only": config_weights(
            global_weights, lambda g, i, a: a[0] == "graph"
        ),
        "harmonic_only": config_weights(
            global_weights, lambda g, i, a: a[0] == "harmonic"
        ),
        "full_no_graph": config_weights(
            global_weights, lambda g, i, a: a[0] != "graph"
        ),
        "full_no_harmonic": config_weights(
            global_weights, lambda g, i, a: a[0] != "harmonic"
        ),
    }
    for group, actions in PORTFOLIOS.items():
        for index, action in enumerate(actions):
            family, parameters, _ = action
            name = f"action_g{group}_a{index}_{family}_{'_'.join(map(str, parameters))}"
            fixed_configs[name] = config_weights(
                global_weights, lambda g, i, a, gg=group, ii=index: g == gg and i == ii
            )

    prepared = []
    approximate_rows = {name: [] for name in fixed_configs}
    approximate_group_rows = {
        name: {group: [] for group in PORTFOLIOS} for name in fixed_configs
    }
    subset_group_rows: dict[int, dict[int, list[float]]] = {
        group: {mask: [] for mask in range(1 << len(actions))}
        for group, actions in PORTFOLIOS.items()
    }
    reconstruction_errors = []

    for fold, row in enumerate(folds):
        val, labels, old_base, logs, reconstruction_error = build_fold_action_logs(
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
            diagnostics,
        )
        reconstruction_errors.append(reconstruction_error)
        anchors = np.setdiff1d(np.arange(len(val)), row["query_local"])
        train_mask = valid.copy()
        train_mask[val] = False
        train_mask[val[anchors]] = True
        train = np.flatnonzero(train_mask)
        query_labels = labels[row["query_local"]]
        calibrated, calibration_diagnostic = calibrated_weights(
            pos,
            row["query"],
            query_labels,
            train,
            official_counts,
            official_log_d1,
        )
        row["calibrated_weights"] = calibrated
        old_profile = aggregate(old_base)
        new_profile = aggregate(row["base"])
        truth_profile = aggregate_target[row["query"]]
        baseline_point = profile_cosine(new_profile, truth_profile)

        for name, weights in fixed_configs.items():
            prediction = approximate_target(
                old_profile,
                new_profile,
                labels,
                row["query_local"],
                logs,
                weights,
                legacy_old_base=name == "full_global_legacy",
            )
            delta = profile_cosine(prediction, truth_profile) - baseline_point
            approximate_rows[name].append(weighted_mean(delta, calibrated))
            for group in PORTFOLIOS:
                group_mask = query_labels == group
                approximate_group_rows[name][group].append(
                    weighted_mean(delta[group_mask], calibrated[group_mask])
                )

        for group, actions in PORTFOLIOS.items():
            query_mask = query_labels == group
            group_weight = calibrated[query_mask]
            for subset in range(1 << len(actions)):
                weights = {
                    key: (
                        value
                        if key[0] == group and (subset & (1 << key[1]))
                        else 0.0
                    )
                    for key, value in global_weights.items()
                }
                prediction = approximate_target(
                    old_profile,
                    new_profile,
                    labels,
                    row["query_local"],
                    logs,
                    weights,
                )
                delta = profile_cosine(prediction, truth_profile) - baseline_point
                subset_group_rows[group][subset].append(
                    weighted_mean(delta[query_mask], group_weight)
                )

        prepared.append(
            {
                "row": row,
                "labels": labels,
                "old_base": old_base,
                "logs": logs,
                "calibration": calibration_diagnostic,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "prepare_action_logs",
                    "fold": fold,
                    "reconstruction_error": reconstruction_error,
                }
            ),
            flush=True,
        )

    def select_subsets(allowed_families: set[str] | None) -> tuple[dict, dict]:
        selected = {}
        weights = global_weights.copy()
        for group, actions in PORTFOLIOS.items():
            rows = []
            for subset, deltas_list in subset_group_rows[group].items():
                if allowed_families is not None and any(
                    (subset & (1 << index))
                    and action[0] not in allowed_families
                    for index, action in enumerate(actions)
                ):
                    continue
                delta = np.asarray(deltas_list)
                rows.append(
                    {
                        "subset": subset,
                        "action_indices": [
                            index
                            for index in range(len(actions))
                            if subset & (1 << index)
                        ],
                        "deltas": delta.tolist(),
                        "mean": float(delta.mean()),
                        "min": float(delta.min()),
                        "lcb": float(delta.mean() - 0.75 * delta.std()),
                    }
                )
            safe = [candidate for candidate in rows if candidate["min"] >= -1e-7]
            best = max(safe or rows, key=lambda item: (item["lcb"], item["mean"]))
            selected[str(group)] = best
            keep = set(best["action_indices"])
            for index in range(len(actions)):
                weights[(group, index)] = (
                    global_weights[(group, index)] if index in keep else 0.0
                )
        return selected, weights

    robust_all_selection, robust_all_weights = select_subsets(None)
    robust_non_density_selection, robust_non_density_weights = select_subsets(
        {"canonical", "anchor", "gp"}
    )
    fixed_configs["robust_binary_all"] = robust_all_weights
    fixed_configs["robust_binary_non_density"] = robust_non_density_weights
    fixed_configs["robust_binary_all_alpha100"] = robust_all_weights
    fixed_configs["robust_binary_all_alpha125"] = robust_all_weights

    # Add approximate results for the two data-driven binary selectors now that
    # their group masks have been frozen across all five folds.
    for name, weights in (
        ("robust_binary_all", robust_all_weights),
        ("robust_binary_non_density", robust_non_density_weights),
    ):
        approximate_rows[name] = []
        approximate_group_rows[name] = {group: [] for group in PORTFOLIOS}
        for item in prepared:
            row = item["row"]
            old_profile = aggregate(item["old_base"])
            new_profile = aggregate(row["base"])
            truth_profile = aggregate_target[row["query"]]
            baseline_point = profile_cosine(new_profile, truth_profile)
            prediction = approximate_target(
                old_profile,
                new_profile,
                item["labels"],
                row["query_local"],
                item["logs"],
                weights,
            )
            delta = profile_cosine(prediction, truth_profile) - baseline_point
            approximate_rows[name].append(
                weighted_mean(delta, row["calibrated_weights"])
            )
            query_labels = item["labels"][row["query_local"]]
            for group in PORTFOLIOS:
                group_mask = query_labels == group
                approximate_group_rows[name][group].append(
                    weighted_mean(
                        delta[group_mask], row["calibrated_weights"][group_mask]
                    )
                )

    approximate_summary = []
    for name, deltas_list in approximate_rows.items():
        delta = np.asarray(deltas_list)
        approximate_summary.append(
            {
                "name": name,
                "fold_c1_deltas": delta.tolist(),
                "geometry_weighted_score_delta_approx": float(
                    0.4 * np.dot(FOLD_WEIGHTS, delta)
                ),
                "min_score_delta_approx": float(0.4 * delta.min()),
                "lcb_score_delta_approx": float(
                    0.4 * (delta.mean() - 0.75 * delta.std())
                ),
                "edge_composite_score_delta_approx": float(
                    0.4
                    * sum(
                        official_counts[group]
                        * approximate_group_rows[name][group][
                            COMPOSITE_FOLD_BY_GROUP[group]
                        ]
                        for group in PORTFOLIOS
                    )
                    / sum(official_counts.values())
                ),
            }
        )
    approximate_summary.sort(
        key=lambda item: min(
            item["geometry_weighted_score_delta_approx"],
            item["edge_composite_score_delta_approx"],
        ),
        reverse=True,
    )

    # Exact full-channel audit is intentionally limited to interpretable family
    # decompositions plus the two frozen robust binary selectors.
    exact_names = [
        "full_global_legacy",
        "full_global",
        "full_safe_weights",
        "density_graph_harmonic",
        "non_density_all",
        "canonical_anchor",
        "canonical_gp",
        "rbf_gp",
        "tree_gp",
        "graph_canonical",
        "graph_canonical_gp",
        "canonical_only",
        "canonical_tree_only",
        "canonical_rbf_only",
        "tree_anchor",
        "rbf_anchor",
        "anchor_only",
        "gp_only",
        "graph_only",
        "harmonic_only",
        "full_no_graph",
        "full_no_harmonic",
        "robust_binary_all",
        "robust_binary_non_density",
        "robust_binary_all_alpha100",
        "robust_binary_all_alpha125",
    ]
    fold_exact_rows = []
    group_accumulators_by_fold = []
    for fold, item in enumerate(prepared):
        row = item["row"]
        labels = item["labels"][row["query_local"]]
        desired = {
            name: exact_target(
                item["old_base"],
                row["base"],
                item["labels"],
                row["query_local"],
                item["logs"],
                fixed_configs[name],
                legacy_old_base=name == "full_global_legacy",
                pas_alpha=(
                    1.0
                    if name.endswith("alpha100")
                    else 1.25
                    if name.endswith("alpha125")
                    else PAS_ALPHA
                ),
            )
            for name in exact_names
        }
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        accumulator = np.zeros((1 + len(exact_names), 6), dtype=np.float64)
        group_accumulator = np.zeros(
            (1 + len(exact_names), 11, 6), dtype=np.float64
        )
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
                row["calibrated_weights"][start:stop].astype(np.float32), device=DEVICE
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
            base_band = torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE)
            for index, name in enumerate(exact_names, 1):
                target_band = torch.as_tensor(
                    desired[name][start:stop].copy(), device=DEVICE
                )
                value = project(
                    p,
                    base_band,
                    target_band,
                    24,
                    PROJECTION_ITERATIONS,
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
        baseline = components(accumulator[0])
        rows = []
        for index, name in enumerate(exact_names, 1):
            value = components(accumulator[index])
            rows.append(
                {
                    "name": name,
                    **value,
                    "delta_vs_phase6": float(value["score"] - baseline["score"]),
                    "component_deltas": {
                        key: float(value[key] - baseline[key])
                        for key in ("c1_pas", "c2_pdp", "c3_nmse")
                    },
                }
            )
        fold_exact_rows.append({"fold": fold, "baseline": baseline, "rows": rows})
        group_accumulators_by_fold.append(group_accumulator)
        print(
            json.dumps(
                {
                    "stage": "exact_fold",
                    "fold": fold,
                    "best": max(rows, key=lambda value: value["delta_vs_phase6"]),
                }
            ),
            flush=True,
        )

    composite_baseline_stats = np.zeros(6, dtype=np.float64)
    for group, fold in COMPOSITE_FOLD_BY_GROUP.items():
        composite_baseline_stats += group_accumulators_by_fold[fold][0, group]
    composite_baseline = components(composite_baseline_stats)
    exact_summary = []
    for index, name in enumerate(exact_names, 1):
        deltas = np.asarray(
            [fold["rows"][index - 1]["delta_vs_phase6"] for fold in fold_exact_rows]
        )
        composite_stats = np.zeros(6, dtype=np.float64)
        for group, fold in COMPOSITE_FOLD_BY_GROUP.items():
            composite_stats += group_accumulators_by_fold[fold][index, group]
        composite = components(composite_stats)
        exact_summary.append(
            {
                "name": name,
                "fold_score_deltas": deltas.tolist(),
                "geometry_weighted_delta": float(np.dot(FOLD_WEIGHTS, deltas)),
                "min_fold_delta": float(deltas.min()),
                "lcb_delta": float(deltas.mean() - 0.75 * deltas.std()),
                "edge_composite_score": composite["score"],
                "edge_composite_delta": float(
                    composite["score"] - composite_baseline["score"]
                ),
                "edge_component_deltas": {
                    key: float(composite[key] - composite_baseline[key])
                    for key in ("c1_pas", "c2_pdp", "c3_nmse")
                },
                "group_fold_score_deltas": {
                    str(group): [
                        float(
                            components(group_accumulators_by_fold[fold][index, group])[
                                "score"
                            ]
                            - components(group_accumulators_by_fold[fold][0, group])[
                                "score"
                            ]
                        )
                        for fold in range(5)
                    ]
                    for group in PORTFOLIOS
                },
                "robust_proxy": float(
                    min(
                        np.dot(FOLD_WEIGHTS, deltas),
                        composite["score"] - composite_baseline["score"],
                    )
                ),
            }
        )

    index_by_name = {name: index for index, name in enumerate(exact_names, 1)}

    def exact_groupwise_selector(
        name: str, allowed_names: list[str]
    ) -> tuple[dict, dict[tuple[int, int], float], dict]:
        selection = {}
        selected_indices = {}
        combined_weights = {key: 0.0 for key in global_weights}
        for group in PORTFOLIOS:
            options = [("baseline", 0)] + [
                (candidate_name, index_by_name[candidate_name])
                for candidate_name in allowed_names
            ]
            rows = []
            for candidate_name, index in options:
                delta = np.asarray(
                    [
                        components(group_accumulators_by_fold[fold][index, group])[
                            "score"
                        ]
                        - components(group_accumulators_by_fold[fold][0, group])["score"]
                        for fold in range(5)
                    ]
                )
                rows.append(
                    {
                        "config": candidate_name,
                        "index": index,
                        "fold_score_deltas": delta.tolist(),
                        "mean": float(delta.mean()),
                        "min": float(delta.min()),
                        "lcb": float(delta.mean() - 0.75 * delta.std()),
                    }
                )
            safe = [candidate for candidate in rows if candidate["min"] >= -1e-10]
            best = max(safe, key=lambda value: (value["lcb"], value["mean"]))
            selection[str(group)] = best
            selected_indices[group] = best["index"]
            if best["config"] != "baseline":
                for index in range(len(PORTFOLIOS[group])):
                    combined_weights[(group, index)] = fixed_configs[best["config"]][
                        (group, index)
                    ]

        fold_deltas = []
        for fold in range(5):
            baseline_stats = sum(
                (
                    group_accumulators_by_fold[fold][0, group]
                    for group in PORTFOLIOS
                ),
                np.zeros(6, dtype=np.float64),
            )
            candidate_stats = sum(
                (
                    group_accumulators_by_fold[fold][selected_indices[group], group]
                    for group in PORTFOLIOS
                ),
                np.zeros(6, dtype=np.float64),
            )
            fold_deltas.append(
                components(candidate_stats)["score"]
                - components(baseline_stats)["score"]
            )
        fold_delta = np.asarray(fold_deltas)
        edge_stats = sum(
            (
                group_accumulators_by_fold[COMPOSITE_FOLD_BY_GROUP[group]][
                    selected_indices[group], group
                ]
                for group in PORTFOLIOS
            ),
            np.zeros(6, dtype=np.float64),
        )
        edge = components(edge_stats)
        summary = {
            "name": name,
            "fold_score_deltas": fold_delta.tolist(),
            "geometry_weighted_delta": float(np.dot(FOLD_WEIGHTS, fold_delta)),
            "min_fold_delta": float(fold_delta.min()),
            "lcb_delta": float(fold_delta.mean() - 0.75 * fold_delta.std()),
            "edge_composite_score": edge["score"],
            "edge_composite_delta": float(
                edge["score"] - composite_baseline["score"]
            ),
            "edge_component_deltas": {
                key: float(edge[key] - composite_baseline[key])
                for key in ("c1_pas", "c2_pdp", "c3_nmse")
            },
            "group_fold_score_deltas": {
                str(group): selection[str(group)]["fold_score_deltas"]
                for group in PORTFOLIOS
            },
            "robust_proxy": float(
                min(
                    np.dot(FOLD_WEIGHTS, fold_delta),
                    edge["score"] - composite_baseline["score"],
                )
            ),
        }
        return selection, combined_weights, summary

    non_density_exact_names = [
        "non_density_all",
        "canonical_anchor",
        "canonical_gp",
        "rbf_gp",
        "tree_gp",
        "canonical_only",
        "canonical_tree_only",
        "canonical_rbf_only",
        "tree_anchor",
        "rbf_anchor",
        "anchor_only",
        "gp_only",
        "robust_binary_non_density",
    ]
    all_exact_names = [
        candidate_name
        for candidate_name in exact_names
        if candidate_name != "full_global_legacy"
    ]
    groupwise_all_selection, groupwise_all_weights, groupwise_all_summary = (
        exact_groupwise_selector("groupwise_exact_all", all_exact_names)
    )
    (
        groupwise_non_density_selection,
        groupwise_non_density_weights,
        groupwise_non_density_summary,
    ) = exact_groupwise_selector(
        "groupwise_exact_non_density", non_density_exact_names
    )
    fixed_configs["groupwise_exact_all"] = groupwise_all_weights
    fixed_configs["groupwise_exact_non_density"] = groupwise_non_density_weights
    exact_summary.extend([groupwise_all_summary, groupwise_non_density_summary])
    exact_summary.sort(
        key=lambda item: (
            item["robust_proxy"],
            item["min_fold_delta"],
            item["geometry_weighted_delta"],
        ),
        reverse=True,
    )

    action_manifest = {
        str(group): [
            {
                "index": index,
                "family": action[0],
                "parameters": action[1],
                "exponent": action[2],
                "global_weight": global_weights[(group, index)],
                "safe_weight": safe_weights[(group, index)],
            }
            for index, action in enumerate(actions)
        ]
        for group, actions in PORTFOLIOS.items()
    }
    config_manifest = {
        name: {
            str(group): [weights[(group, index)] for index in range(len(actions))]
            for group, actions in PORTFOLIOS.items()
        }
        for name, weights in fixed_configs.items()
        if name in exact_names or name.startswith("groupwise_exact")
    }
    output = {
        "pas_alpha": PAS_ALPHA,
        "projection_iterations": PROJECTION_ITERATIONS,
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "composite_fold_by_group": COMPOSITE_FOLD_BY_GROUP,
        "reconstruction_max_errors": reconstruction_errors,
        "action_manifest": action_manifest,
        "robust_binary_all_selection": robust_all_selection,
        "robust_binary_non_density_selection": robust_non_density_selection,
        "groupwise_exact_all_selection": groupwise_all_selection,
        "groupwise_exact_non_density_selection": groupwise_non_density_selection,
        "config_manifest": config_manifest,
        "approximate_summary": approximate_summary,
        "fold_exact": fold_exact_rows,
        "edge_composite_baseline": composite_baseline,
        "exact_summary": exact_summary,
    }
    (ROOT / "phase10_phase7_component_decompose.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "summary", "top": exact_summary[:10]}), flush=True)


if __name__ == "__main__":
    run()
