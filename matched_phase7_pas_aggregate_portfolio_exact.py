from __future__ import annotations

import json
import itertools
import os
from pathlib import Path

import numpy as np
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
from matched_phase7_pas_harmonic_graph import harmonic_coefficients


ROOT = Path(__file__).resolve().parent
WEIGHT_OPTIONS = (0.0, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)

# Actions are (family, parameters, exponent). The exponent is the scale that
# was independently validated for that action before portfolio construction.
PORTFOLIOS = {
    0: [("graph", ("plain", "map8_w2", 6, "inverse2"), 0.025)],
    1: [
        ("harmonic", ("plain", 8, "gauss20"), 0.15),
        ("graph", ("h", "polar50", 6, "exp20"), 0.15),
        ("canonical", ("hv_forest3",), 0.15),
    ],
    2: [
        ("graph", ("plain", "map8_w2", 12, "inverse2"), 0.10),
        ("anchor", ("plain", "mean"), 0.10),
        ("canonical", ("plain_extra3",), 0.05),
    ],
    3: [
        ("canonical", ("h_extra8",), 0.15),
        ("anchor", ("plain", "idw1"), 0.075),
        ("graph", ("h", "map16_w4", 6, "exp20"), 0.15),
    ],
    4: [
        ("graph", ("h", "map16_w4", 8, "inverse2"), 0.10),
        ("gp", ("plain", 64, "matern32", 2.0, 0.15, "ordinary"), 0.05),
        ("anchor", ("plain", "nearest"), 0.025),
    ],
    5: [
        ("graph", ("hv", "map16_w4", 6, "exp20"), 0.025),
        ("anchor", ("hv", "mean"), 0.10),
        ("gp", ("hv", 32, "matern32", 1.0, 0.03, "ordinary"), 0.025),
    ],
    6: [
        ("harmonic", ("plain", 6, "gauss20"), 0.15),
        ("anchor", ("plain", "nearest"), 0.10),
        ("canonical", ("plain_rbf",), 0.15),
    ],
    7: [
        ("canonical", ("hv_rbf",), 0.10),
        ("anchor", ("plain", "mean"), 0.075),
        ("graph", ("h", "map16_w4", 6, "exp20"), 0.025),
    ],
    8: [
        ("graph", ("hv", "xy_x025", 6, "inverse2"), 0.15),
        ("anchor", ("plain", "idw3"), 0.075),
        ("harmonic", ("hv", 20, "inverse2"), 0.015),
    ],
    9: [
        ("graph", ("hv", "xy_x4", 8, "inverse2"), 0.15),
        ("gp", ("h", 32, "exponential", 0.5, 0.03, "gp"), 0.30),
        ("canonical", ("hv_rbf",), 0.15),
    ],
    10: [
        ("graph", ("hv", "map16_w4", 6, "exp20"), 0.15),
        ("gp", ("plain", 32, "matern32", 0.5, 0.03, "ordinary"), 0.20),
        ("anchor", ("hv", "nearest"), 0.15),
    ],
}


def normalize_bs(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def run() -> None:
    pos, _, energy = rp.load_data()
    valid_mask = energy > 0
    valid = np.flatnonzero(valid_mask)
    target = build_cache()
    aggregate_target = aggregate(target)
    unit, side = direction(pos)
    horizontal_moment = h_moment(target, valid)
    vertical_moment = v_moment(target, valid)
    coordinates = build_coordinates(pos, valid)
    diagnostics = json.loads((ROOT / "matched_rect_split_diagnostics.json").read_text())
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    global_selection_path = ROOT / "matched_phase7_pas_aggregate_global_selection.json"
    global_selection = None
    if global_selection_path.exists():
        selection_payload = json.loads(global_selection_path.read_text())
        global_selection = next(
            row
            for row in selection_payload["results"]
            if row["mode"] == "lcb" and float(row["penalty"]) == 0.0
        )["selected"]
    fold_rows = []

    for fold in range(5):
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
        base = np.asarray(
            np.load(
                ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
                mmap_mode="r",
            )
        )
        truth = np.asarray(target[val])
        base_profile = aggregate(base)
        baseline_point = point_cosine(base, truth)
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
                random_state=seed_base + 10 * fold + ("plain", "h", "hv").index(variant),
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
            v_val = (
                vertical_shift[val]
                if variant == "hv"
                else np.zeros(len(val), int)
            )
            return normalize_profile(
                np.maximum(roll_profile(value, h_val, v_val), 0) ** 2
            ).astype(np.float32)

        def candidate(family: str, parameters: tuple, group: int) -> np.ndarray:
            key = (family, parameters, group if family == "anchor" else -1)
            if key in cache:
                return cache[key]
            if family == "canonical":
                name = parameters[0]
                value = np.load(
                    ROOT / f"matched_phase7_aggregate_{name}_fold{fold}.npy",
                    mmap_mode="r",
                )
                output = np.asarray(value)
            elif family in ("harmonic", "graph"):
                if family == "harmonic":
                    variant, neighbors, kernel = parameters
                    node_coordinates = pos[np.concatenate((train, val)), :2].astype(np.float32)
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
                output = base_profile.copy()
                item = next(
                    row
                    for row in diagnostics
                    if row["fold"] == fold and row["group"] == group
                )
                lo, hi = np.asarray(item["box_lo"]), np.asarray(item["box_hi"])
                inside = valid_mask & np.all(pos[:, :2] >= lo, axis=1) & np.all(
                    pos[:, :2] <= hi, axis=1
                )
                anchors = np.flatnonzero(
                    inside & ~np.isin(np.arange(len(pos)), val)
                )
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
                    source = roll_profile(
                        aggregate_target[anchors], h_anchor, v_anchor
                    )
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

        log_ratio_by_group = {}
        epsilon = 1e-3 / base.shape[1]
        for group, actions in PORTFOLIOS.items():
            mask = labels == group
            current = base_profile[mask]
            action_log_ratios = []
            for family, parameters, scale in actions:
                desired = candidate(family, parameters, group)[mask]
                ratio = np.clip((desired + epsilon) / (current + epsilon), 0.25, 4.0)
                action_log_ratios.append(scale * np.log(ratio))
            log_ratio_by_group[group] = action_log_ratios

        if global_selection is not None:
            desired_global = base.copy()
            for group, action_log_ratios in log_ratio_by_group.items():
                mask = labels == group
                action_weights = global_selection[str(group)]["action_weights"]
                log_ratio = sum(
                    weight * value
                    for weight, value in zip(action_weights, action_log_ratios)
                )
                desired_global[mask] = normalize_bs(
                    base[mask] * np.exp(log_ratio)[:, :, None, None]
                )
            np.save(
                ROOT
                / f"matched_phase7_aggregate_portfolio_mean_pas_band24_fold{fold}.npy",
                desired_global.astype(np.float32),
            )

        group_deltas = {group: {} for group in PORTFOLIOS}
        for group, action_log_ratios in log_ratio_by_group.items():
            mask = labels == group
            for action_weights in itertools.product(
                WEIGHT_OPTIONS, repeat=len(action_log_ratios)
            ):
                if not any(action_weights):
                    continue
                log_ratio = sum(
                    weight * value
                    for weight, value in zip(action_weights, action_log_ratios)
                )
                prediction = normalize_bs(
                    base[mask] * np.exp(log_ratio)[:, :, None, None]
                )
                delta_point = (
                    point_cosine(prediction, truth[mask]) - baseline_point[mask]
                )
                group_deltas[group][action_weights] = float(delta_point.mean())
        fold_rows.append(group_deltas)
        print(
            json.dumps({"stage": "fold", "fold": fold, "cache": len(cache)}),
            flush=True,
        )

    groups = {}
    selected = {}
    for group in PORTFOLIOS:
        rows = []
        for action_weights in itertools.product(
            WEIGHT_OPTIONS, repeat=len(PORTFOLIOS[group])
        ):
            if not any(action_weights):
                continue
            delta = np.asarray(
                [fold_rows[fold][group][action_weights] for fold in range(5)]
            )
            rows.append(
                {
                    "action_weights": action_weights,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
        rows.sort(key=lambda row: row["lcb"], reverse=True)
        groups[str(group)] = rows
        safe = [row for row in rows if row["min_delta"] > 0]
        selected[str(group)] = safe[0] if safe else {
            "action_weights": [0.0] * len(PORTFOLIOS[group]),
            "deltas": [0.0] * 5,
            "mean_delta": 0.0,
            "min_delta": 0.0,
            "lcb": 0.0,
        }
    combined = np.zeros(5)
    for group, count in counts.items():
        combined += count * np.asarray(selected[str(int(group))]["deltas"])
    combined /= sum(counts.values())
    payload = {
        "portfolios": PORTFOLIOS,
        "groups": groups,
        "selected": selected,
        "combined": {
            "c1_deltas": combined.tolist(),
            "score_deltas_approx": (0.4 * combined).tolist(),
            "mean_score_delta_approx": float(0.4 * combined.mean()),
            "min_score_delta_approx": float(0.4 * combined.min()),
        },
    }
    (ROOT / "matched_phase7_pas_aggregate_portfolio_exact.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(json.dumps({"selected": selected, "combined": payload["combined"]}), flush=True)


if __name__ == "__main__":
    run()
