from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
from phase8_anchor_local_gate_channel_validation import ALPHA_GRID, components, prepare_folds
from phase8_anchor_retained_pas_resolution_validation import project
from phase8_anchor_retained_pas_screen import horizontal_shifts, mapped_anchors, normalize
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_buildable_residual_gate_pas_screen import external_residual
from phase10_calibrated_pas_residual_joint_validation import update_weighted_scores


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
ANCHOR_GROUPS = (4, 5, 10)
RESIDUAL_ALPHAS = (0.05, 0.10, 0.15)
LOCAL_SCALES = (0.00, 0.25, 0.50)
CONFIGS = tuple((r, l) for r in RESIDUAL_ALPHAS for l in LOCAL_SCALES)
PRIMARY_CONFIG = (0.10, 0.50)
AGGREGATE_SCALE = 0.75
ITERATIONS = 4
FOLD_WEIGHTS = np.asarray([0.312, 0.357, 0.229, 0.046, 0.057], dtype=np.float64)
FOLD_WEIGHTS /= FOLD_WEIGHTS.sum()
LOCKED_FOLD_BY_GROUP = {0: 1, 1: 1, 2: 0, 3: 1, 4: 2, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 2}


def calibrated_weights(
    pos: np.ndarray,
    query: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    official_counts: dict[int, int],
    official_log_d1: dict[int, float],
) -> tuple[np.ndarray, dict[str, object]]:
    distance = np.asarray(cKDTree(pos[train, :2]).query(pos[query, :2], k=1)[0])
    z_all = np.log1p(distance)
    output = np.zeros(len(query), dtype=np.float64)
    diagnostics: dict[str, object] = {}
    for group in sorted(official_counts):
        rows = np.flatnonzero(labels == group)
        if len(rows) == 0:
            continue
        z = z_all[rows]
        if group == 2:
            multiplier = np.ones(len(rows), dtype=np.float64)
        else:
            center = float(np.mean(z))
            lo, hi = -20.0, 20.0
            for _ in range(80):
                value = 0.5 * (lo + hi)
                multiplier = np.exp(np.clip(value * (z - center), -30.0, 30.0))
                multiplier /= np.mean(multiplier)
                multiplier = np.clip(multiplier, 0.4, 2.5)
                multiplier /= np.mean(multiplier)
                if float(np.average(z, weights=multiplier)) < official_log_d1[group]:
                    lo = value
                else:
                    hi = value
            multiplier = np.exp(
                np.clip(0.5 * (lo + hi) * (z - center), -30.0, 30.0)
            )
            multiplier /= np.mean(multiplier)
            multiplier = np.clip(multiplier, 0.4, 2.5)
            multiplier /= np.mean(multiplier)
        output[rows] = official_counts[group] / len(rows) * multiplier
        diagnostics[str(group)] = {
            "n": int(len(rows)),
            "d1_quantiles": np.quantile(distance[rows], [0.1, 0.5, 0.9]).tolist(),
            "log_d1_mean_before": float(np.mean(z)),
            "log_d1_mean_after": float(np.average(z, weights=multiplier)),
            "log_d1_mean_official": float(official_log_d1[group]),
            "multiplier_range": [float(multiplier.min()), float(multiplier.max())],
        }
    return output, diagnostics


def composite_components(
    group_accumulators: list[np.ndarray], index: int, mapping: dict[int, int]
) -> dict[str, float]:
    total = np.zeros(6, dtype=np.float64)
    for group, fold in mapping.items():
        total += group_accumulators[fold][index, group]
    return components(total)


@torch.no_grad()
def run() -> None:
    folds, pos, channel, energy, official_counts, actual_counts = prepare_folds()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    valid = energy > 0
    official_d1 = np.asarray(cKDTree(pos[valid, :2]).query(test_pos[:, :2], k=1)[0])
    official_log_d1 = {
        int(group): float(np.mean(np.log1p(official_d1[test_labels == group])))
        for group in np.unique(test_labels)
    }
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)
    descriptor = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    shifts = horizontal_shifts(pos)
    diagnostics = []
    for fold, row in enumerate(folds):
        val = row["val"]
        labels = row["labels"]
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        external_mask = valid.copy()
        external_mask[val] = False
        external_train = np.flatnonzero(external_mask)
        row["residual"] = external_residual(
            pos,
            val,
            labels,
            anchors,
            row["query_local"],
            descriptor,
            external_train,
            shifts,
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
        diagnostics.append(diagnostic)
        print(json.dumps({"stage": "prepare_calibrated", "fold": fold}), flush=True)

    gate_alpha = []
    for heldout in range(5):
        train_x = np.concatenate([folds[i]["x"] for i in range(5) if i != heldout])
        train_y = np.concatenate([folds[i]["gain_grid"] for i in range(5) if i != heldout])
        train_w = np.concatenate(
            [folds[i]["calibrated_weights"] for i in range(5) if i != heldout]
        )
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=80,
            max_features=0.7,
            n_jobs=-1,
            random_state=52180,
        )
        model.fit(train_x, train_y, sample_weight=train_w)
        gate_alpha.append(ALPHA_GRID[np.argmax(model.predict(folds[heldout]["x"]), axis=1)])

    # Index 0 is Phase6 and index 1 is the frozen Phase7 PAS-only ablation.
    group_accumulators: list[np.ndarray] = []
    fold_rows = []
    for fold, row in enumerate(folds):
        labels = row["labels"][row["query_local"]]
        aggregate_all = np.load(
            ROOT / f"matched_phase7_aggregate_portfolio_mean_pas_band24_fold{fold}.npy",
            mmap_mode="r",
        )
        aggregate = np.asarray(aggregate_all[row["query_local"]])
        phase7_target = normalize(
            (1.0 - AGGREGATE_SCALE) * row["base"] + AGGREGATE_SCALE * aggregate
        ).astype(np.float32)
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        accumulator = np.zeros((2 + len(CONFIGS), 6), dtype=np.float64)
        group_accumulator = np.zeros((2 + len(CONFIGS), 11, 6), dtype=np.float64)
        mask_all = np.isin(labels, ANCHOR_GROUPS).astype(np.float32)[:, None, None, None]
        desired_by_config = {}
        for config in CONFIGS:
            residual_alpha, local_scale = config
            corrected = normalize(
                phase7_target * np.exp(residual_alpha * mask_all * row["residual"])
            )
            alpha = mask_all * np.clip(
                local_scale * gate_alpha[fold], 0.0, 0.30
            )[:, None, None, None]
            desired_by_config[config] = normalize(
                (1.0 - alpha) * corrected + alpha * row["local"]
            ).astype(np.float32)

        for start in range(0, len(row["query"]), 2):
            stop = min(start + 2, len(row["query"]))
            local_query = row["query_local"][start:stop]
            p = torch.as_tensor(np.asarray(prediction[local_query]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(channel[row["query"][start:stop]]).copy(), device=DEVICE)
            weights = torch.as_tensor(
                row["calibrated_weights"][start:stop].astype(np.float32), device=DEVICE
            )
            truth_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            truth_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
            base_band = torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE)
            batch_labels = labels[start:stop]
            update_weighted_scores(
                accumulator, group_accumulator, 0, p, t, truth_pas, truth_pdp,
                weights, batch_labels,
            )
            phase7_value = project(
                p,
                base_band,
                torch.as_tensor(phase7_target[start:stop].copy(), device=DEVICE),
                24,
                ITERATIONS,
            )
            update_weighted_scores(
                accumulator, group_accumulator, 1, phase7_value, t, truth_pas,
                truth_pdp, weights, batch_labels,
            )
            for index, config in enumerate(CONFIGS, 2):
                desired = torch.as_tensor(
                    desired_by_config[config][start:stop].copy(), device=DEVICE
                )
                value = project(p, base_band, desired, 24, ITERATIONS)
                update_weighted_scores(
                    accumulator, group_accumulator, index, value, t, truth_pas,
                    truth_pdp, weights, batch_labels,
                )
        base_components = components(accumulator[0])
        pas_components = components(accumulator[1])
        rows = []
        for index, config in enumerate(CONFIGS, 2):
            value = components(accumulator[index])
            rows.append(
                {
                    "residual_alpha": config[0],
                    "local_scale": config[1],
                    **value,
                    "delta_vs_phase6": value["score"] - base_components["score"],
                    "anchor_increment_vs_pas_only": value["score"] - pas_components["score"],
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "phase6": base_components,
                "pas_only": {
                    **pas_components,
                    "delta_vs_phase6": pas_components["score"] - base_components["score"],
                },
                "rows": rows,
            }
        )
        group_accumulators.append(group_accumulator)
        print(
            json.dumps({"stage": "fold", "fold": fold, "primary": rows[CONFIGS.index(PRIMARY_CONFIG)]}),
            flush=True,
        )

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
    composite_baselines = {
        name: composite_components(group_accumulators, 0, mapping)
        for name, mapping in mappings.items()
    }
    composite_pas = {
        name: composite_components(group_accumulators, 1, mapping)
        for name, mapping in mappings.items()
    }
    summary = []
    for index, config in enumerate(CONFIGS, 2):
        fold_delta = np.asarray(
            [fold["rows"][index - 2]["delta_vs_phase6"] for fold in fold_rows]
        )
        fold_anchor = np.asarray(
            [fold["rows"][index - 2]["anchor_increment_vs_pas_only"] for fold in fold_rows]
        )
        composite_rows = {}
        for name, mapping in mappings.items():
            value = composite_components(group_accumulators, index, mapping)
            composite_rows[name] = {
                **value,
                "delta_vs_phase6": value["score"] - composite_baselines[name]["score"],
                "anchor_increment_vs_pas_only": value["score"] - composite_pas[name]["score"],
            }
        inner_anchor = np.asarray(
            [composite_rows[f"inner_{rotation}"]["anchor_increment_vs_pas_only"] for rotation in range(1, 5)]
        )
        summary.append(
            {
                "residual_alpha": config[0],
                "local_scale": config[1],
                "is_predeclared_primary": config == PRIMARY_CONFIG,
                "fold_deltas_vs_phase6": fold_delta.tolist(),
                "fold_anchor_increment_vs_pas_only": fold_anchor.tolist(),
                "geometry_weighted_delta_vs_phase6": float(np.dot(FOLD_WEIGHTS, fold_delta)),
                "all_fold_min_delta_vs_phase6": float(fold_delta.min()),
                "inner_anchor_increments": inner_anchor.tolist(),
                "inner_anchor_positive_count": int(np.sum(inner_anchor > 0)),
                "inner_anchor_mean": float(inner_anchor.mean()),
                "composites": composite_rows,
            }
        )
    # This sort is diagnostic only.  The locked composite is not used to select a
    # config; PRIMARY_CONFIG was declared before this evaluation.
    summary.sort(
        key=lambda item: (
            item["inner_anchor_positive_count"],
            item["inner_anchor_mean"],
            item["geometry_weighted_delta_vs_phase6"],
        ),
        reverse=True,
    )
    output = {
        "aggregate_scale": AGGREGATE_SCALE,
        "anchor_groups": ANCHOR_GROUPS,
        "iterations": ITERATIONS,
        "primary_config": PRIMARY_CONFIG,
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "locked_fold_by_group": LOCKED_FOLD_BY_GROUP,
        "official_log_d1": official_log_d1,
        "diagnostics": diagnostics,
        "composite_phase6": composite_baselines,
        "composite_pas_only": composite_pas,
        "folds": fold_rows,
        "summary": summary,
    }
    (ROOT / "phase10_calibrated_anchor_on_pas_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    primary = next(item for item in summary if item["is_predeclared_primary"])
    print(json.dumps({"stage": "summary", "primary": primary, "ranked": summary}), flush=True)


if __name__ == "__main__":
    run()
