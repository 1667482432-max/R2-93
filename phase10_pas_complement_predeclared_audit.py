from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import r2_pipeline as rp
from matched_phase5_tree_band_descriptor import build_cache
from matched_phase6_pas_fitted_canonical import direction, h_moment
from matched_phase6_pas_fitted_vertical import v_moment
from matched_phase7_pas_aggregate_canonical import aggregate
from matched_phase7_pas_aggregate_graph_metric import build_coordinates
from matched_phase7_pas_aggregate_portfolio_exact import PORTFOLIOS
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
from phase10_phase7_component_decompose import (
    action_weight_map,
    build_fold_action_logs,
    config_weights,
    load_selections,
)


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
CORE_GROUPS = {1, 3, 4, 9, 10}
COMPLEMENT_GROUPS = set(range(11)) - CORE_GROUPS
CORE_ALPHA = 1.25
COMPLEMENT_ALPHAS = (0.0, 0.5, 0.75, 1.0)
PROJECTION_ITERATIONS = 4


def selected_robust_weights(global_weights: dict[tuple[int, int], float]) -> dict:
    payload = json.loads(
        (ROOT / "phase10_phase7_component_decompose.json").read_text()
    )
    output = {key: 0.0 for key in global_weights}
    for group in PORTFOLIOS:
        keep = set(
            payload["robust_binary_all_selection"][str(group)]["action_indices"]
        )
        for index in range(len(PORTFOLIOS[group])):
            if index in keep:
                output[(group, index)] = global_weights[(group, index)]
    return output


def mixed_target(
    new_base: np.ndarray,
    labels: np.ndarray,
    query_local: np.ndarray,
    logs: dict[tuple[int, int], np.ndarray],
    core_weights: dict[tuple[int, int], float],
    complement_weights: dict[tuple[int, int], float],
    complement_alpha: float,
) -> np.ndarray:
    output = new_base.copy()
    query_labels = labels[query_local]
    for group, actions in PORTFOLIOS.items():
        query_rows = np.flatnonzero(query_labels == group)
        if not len(query_rows):
            continue
        full_rows = np.flatnonzero(labels == group)
        keep = np.isin(full_rows, query_local)
        weights = core_weights if group in CORE_GROUPS else complement_weights
        alpha = CORE_ALPHA if group in CORE_GROUPS else complement_alpha
        total = sum(
            weights[(group, index)] * logs[(group, index)][keep]
            for index in range(len(actions))
        )
        desired = normalize(
            new_base[query_rows] * np.exp(total)[:, :, None, None]
        )
        output[query_rows] = normalize(
            (1.0 - alpha) * new_base[query_rows] + alpha * desired
        )
    return output.astype(np.float32)


@torch.no_grad()
def run() -> None:
    folds, pos, channel, energy, official_counts, actual_counts = prepare_folds()
    target = build_cache()
    aggregate_target = aggregate(target)
    valid = energy > 0
    valid_index = np.flatnonzero(valid)
    unit, side = direction(pos)
    horizontal_moment = h_moment(target, valid_index)
    vertical_moment = v_moment(target, valid_index)
    coordinates = build_coordinates(pos, valid_index)
    diagnostics = json.loads(
        (ROOT / "matched_rect_split_diagnostics.json").read_text()
    )

    global_selection, _ = load_selections()
    global_weights = action_weight_map(global_selection)
    core_weights = selected_robust_weights(global_weights)
    complement_weights = config_weights(
        global_weights,
        lambda group, index, action: action[0] in {"graph", "canonical", "gp"},
    )

    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    official_distance = np.asarray(
        cKDTree(pos[valid, :2]).query(test_pos[:, :2], k=1)[0]
    )
    official_log_d1 = {
        int(group): float(
            np.mean(np.log1p(official_distance[test_labels == group]))
        )
        for group in np.unique(test_labels)
    }

    names = [f"complement_alpha_{alpha:g}" for alpha in COMPLEMENT_ALPHAS]
    all_group_accumulators = []
    fold_rows = []
    reconstruction_errors = []
    calibration_diagnostics = []

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
        calibration_diagnostics.append(calibration_diagnostic)

        desired = {
            name: mixed_target(
                row["base"],
                labels,
                row["query_local"],
                logs,
                core_weights,
                complement_weights,
                alpha,
            )
            for name, alpha in zip(names, COMPLEMENT_ALPHAS)
        }
        prediction = np.load(
            ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r"
        )
        accumulator = np.zeros((1 + len(names), 6), dtype=np.float64)
        group_accumulator = np.zeros(
            (1 + len(names), 11, 6), dtype=np.float64
        )
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
                calibrated[start:stop].astype(np.float32), device=DEVICE
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
                query_labels[start:stop],
            )
            base_band = torch.as_tensor(
                row["base"][start:stop].copy(), device=DEVICE
            )
            for index, name in enumerate(names, 1):
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
                    query_labels[start:stop],
                )

        baseline = components(accumulator[0])
        candidate_rows = []
        for index, (name, alpha) in enumerate(zip(names, COMPLEMENT_ALPHAS), 1):
            value = components(accumulator[index])
            candidate_rows.append(
                {
                    "name": name,
                    "complement_alpha": alpha,
                    **value,
                    "delta_vs_phase6": float(value["score"] - baseline["score"]),
                    "group_deltas_vs_phase6": {
                        str(group): float(
                            components(group_accumulator[index, group])["score"]
                            - components(group_accumulator[0, group])["score"]
                        )
                        for group in range(11)
                    },
                }
            )
        fold_rows.append(
            {"fold": fold, "baseline": baseline, "rows": candidate_rows}
        )
        all_group_accumulators.append(group_accumulator)
        print(
            json.dumps(
                {
                    "stage": "exact_fold",
                    "fold": fold,
                    "deltas": {
                        row["name"]: row["delta_vs_phase6"]
                        for row in candidate_rows
                    },
                }
            ),
            flush=True,
        )

    composite_baseline_stats = sum(
        (
            all_group_accumulators[fold][0, group]
            for group, fold in COMPOSITE_FOLD_BY_GROUP.items()
        ),
        np.zeros(6, dtype=np.float64),
    )
    composite_baseline = components(composite_baseline_stats)
    summary = []
    robust_deltas = np.asarray(
        [fold["rows"][0]["delta_vs_phase6"] for fold in fold_rows]
    )
    robust_edge_stats = sum(
        (
            all_group_accumulators[fold][1, group]
            for group, fold in COMPOSITE_FOLD_BY_GROUP.items()
        ),
        np.zeros(6, dtype=np.float64),
    )
    robust_edge_delta = (
        components(robust_edge_stats)["score"] - composite_baseline["score"]
    )

    for index, (name, alpha) in enumerate(zip(names, COMPLEMENT_ALPHAS), 1):
        deltas = np.asarray(
            [fold["rows"][index - 1]["delta_vs_phase6"] for fold in fold_rows]
        )
        edge_stats = sum(
            (
                all_group_accumulators[fold][index, group]
                for group, fold in COMPOSITE_FOLD_BY_GROUP.items()
            ),
            np.zeros(6, dtype=np.float64),
        )
        edge = components(edge_stats)
        edge_delta = edge["score"] - composite_baseline["score"]
        increment = deltas - robust_deltas
        geometry_delta = float(np.dot(FOLD_WEIGHTS, deltas))
        geometry_increment = float(np.dot(FOLD_WEIGHTS, increment))
        edge_increment = float(edge_delta - robust_edge_delta)
        summary.append(
            {
                "name": name,
                "complement_alpha": alpha,
                "fold_score_deltas_vs_phase6": deltas.tolist(),
                "fold_increments_vs_robust": increment.tolist(),
                "geometry_weighted_delta_vs_phase6": geometry_delta,
                "geometry_weighted_increment_vs_robust": geometry_increment,
                "edge_composite_delta_vs_phase6": float(edge_delta),
                "edge_composite_increment_vs_robust": edge_increment,
                "robust_increment_proxy": float(
                    min(geometry_increment, edge_increment)
                ),
                "all_fold_total_positive": bool(np.all(deltas > 0)),
                "all_fold_increment_positive": bool(np.all(increment > 0)),
                "passes_predeclared_gate": bool(
                    min(geometry_increment, edge_increment) >= 0.0007
                    and np.all(increment > 0)
                ),
            }
        )

    output = {
        "predeclared_design": {
            "core_groups": sorted(CORE_GROUPS),
            "core_family": "robust_binary_all fixed subset",
            "core_alpha": CORE_ALPHA,
            "complement_groups": sorted(COMPLEMENT_GROUPS),
            "complement_family": "graph_canonical_gp using frozen global weights",
            "complement_alphas": list(COMPLEMENT_ALPHAS),
            "projection_iterations": PROJECTION_ITERATIONS,
            "pass_gate": "min(geometry increment, edge increment) >= 0.0007 and every fold increment > 0",
        },
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "composite_fold_by_group": COMPOSITE_FOLD_BY_GROUP,
        "reconstruction_max_errors": reconstruction_errors,
        "calibration_diagnostics": calibration_diagnostics,
        "folds": fold_rows,
        "summary": summary,
    }
    (ROOT / "phase10_pas_complement_predeclared_audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "summary", "rows": summary}), flush=True)


if __name__ == "__main__":
    run()
