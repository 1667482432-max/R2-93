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
from phase8_anchor_retained_pas_screen import mapped_anchors, normalize
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_buildable_residual_gate_pas_screen import external_residual


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
ALL_GROUPS = tuple(range(11))
NO_GROUPS: tuple[int, ...] = ()
NO_5_8 = (0, 1, 2, 3, 4, 6, 7, 9, 10)
SAFE_ANCHOR = (4, 5, 10)
FOLD_WEIGHTS = np.asarray([0.312, 0.357, 0.229, 0.046, 0.057], dtype=np.float64)
FOLD_WEIGHTS /= FOLD_WEIGHTS.sum()
COMPOSITE_FOLD_BY_GROUP = {0: 1, 1: 1, 2: 0, 3: 1, 4: 2, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 2}


# Candidate choices are intentionally small and hypothesis-driven.  The aggregate
# branch is the Phase-7 PAS signal without its risky PDP branch.  The residual
# branch is the robust Phase-9 PAS signal without its PDP correction.
CANDIDATES = (
    ("aggregate_075_all_i4", 0.75, 0.00, ALL_GROUPS, NO_GROUPS, 4),
    ("aggregate_075_no58_i4", 0.75, 0.00, NO_5_8, NO_GROUPS, 4),
    ("residual_075_all_i12", 0.00, 0.75, NO_GROUPS, ALL_GROUPS, 12),
    ("residual_100_all_i12", 0.00, 1.00, NO_GROUPS, ALL_GROUPS, 12),
    ("residual_075_safe_i12", 0.00, 0.75, NO_GROUPS, SAFE_ANCHOR, 12),
    ("joint_050_050_all_i4", 0.50, 0.50, ALL_GROUPS, ALL_GROUPS, 4),
    ("joint_075_050_all_i4", 0.75, 0.50, ALL_GROUPS, ALL_GROUPS, 4),
    ("joint_050_075_all_i4", 0.50, 0.75, ALL_GROUPS, ALL_GROUPS, 4),
    ("joint_075_075_all_i4", 0.75, 0.75, ALL_GROUPS, ALL_GROUPS, 4),
    ("joint_075_075_all_i12", 0.75, 0.75, ALL_GROUPS, ALL_GROUPS, 12),
    ("joint_075_050_no58_all_i4", 0.75, 0.50, NO_5_8, ALL_GROUPS, 4),
    ("joint_075_075_no58_safe_i4", 0.75, 0.75, NO_5_8, SAFE_ANCHOR, 4),
    ("joint_075_075_all_safe_i4", 0.75, 0.75, ALL_GROUPS, SAFE_ANCHOR, 4),
    ("joint_050_075_all_safe_i4", 0.50, 0.75, ALL_GROUPS, SAFE_ANCHOR, 4),
)


def normalized(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(1e-30)


def difficulty_multiplier(z: np.ndarray, target_mean: float) -> np.ndarray:
    """Continuously tilt a group to the official nearest-neighbour difficulty."""
    if len(z) == 0:
        return np.empty(0, dtype=np.float64)
    center = float(np.mean(z))
    lo, hi = -20.0, 20.0
    for _ in range(80):
        value = 0.5 * (lo + hi)
        multiplier = np.exp(np.clip(value * (z - center), -30.0, 30.0))
        multiplier /= np.mean(multiplier)
        multiplier = np.clip(multiplier, 0.4, 2.5)
        multiplier /= np.mean(multiplier)
        achieved = float(np.average(z, weights=multiplier))
        if achieved < target_mean:
            lo = value
        else:
            hi = value
    multiplier = np.exp(
        np.clip(0.5 * (lo + hi) * (z - center), -30.0, 30.0)
    )
    multiplier /= np.mean(multiplier)
    multiplier = np.clip(multiplier, 0.4, 2.5)
    return multiplier / np.mean(multiplier)


def calibrated_weights(
    pos: np.ndarray,
    query: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    official_counts: dict[int, int],
    official_log_d1: dict[int, float],
) -> tuple[np.ndarray, dict[str, object]]:
    distance = np.asarray(cKDTree(pos[train, :2]).query(pos[query, :2], k=1)[0])
    z = np.log1p(distance)
    output = np.zeros(len(query), dtype=np.float64)
    diagnostics: dict[str, object] = {}
    for group in sorted(official_counts):
        rows = np.flatnonzero(labels == group)
        if len(rows) == 0:
            continue
        multiplier = difficulty_multiplier(z[rows], official_log_d1[group])
        output[rows] = official_counts[group] / len(rows) * multiplier
        diagnostics[str(group)] = {
            "n": int(len(rows)),
            "d1_quantiles": np.quantile(distance[rows], [0.1, 0.5, 0.9]).tolist(),
            "log_d1_mean_before": float(np.mean(z[rows])),
            "log_d1_mean_after": float(np.average(z[rows], weights=multiplier)),
            "log_d1_mean_official": float(official_log_d1[group]),
            "multiplier_range": [float(multiplier.min()), float(multiplier.max())],
        }
    return output, diagnostics


def update_weighted_scores(
    accumulator: np.ndarray,
    group_accumulator: np.ndarray,
    index: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    weights: torch.Tensor,
    labels: np.ndarray,
) -> None:
    pas = torch.abs(rp.bs_fft_torch(prediction)) ** 2
    pas_cos = (pas * target_pas).sum(1) / (
        torch.linalg.vector_norm(pas, dim=1)
        * torch.linalg.vector_norm(target_pas, dim=1)
    ).clamp_min(1e-30)
    pdp = torch.abs(torch.fft.fft(prediction, dim=-1, norm="ortho")) ** 2
    pdp_cos = (pdp * target_pdp).sum(-1) / (
        torch.linalg.vector_norm(pdp, dim=-1)
        * torch.linalg.vector_norm(target_pdp, dim=-1)
    ).clamp_min(1e-30)
    error = torch.abs(prediction - target) ** 2
    target_energy = torch.abs(target) ** 2

    def add(row: np.ndarray, local: torch.Tensor) -> None:
        weight = weights[local]
        row[0] += float((pas_cos[local] * weight[:, None, None]).sum())
        row[1] += float((pdp_cos[local] * weight[:, None, None]).sum())
        row[2] += float(
            (error[local] * weight[:, None, None, None]).sum(dtype=torch.float64)
        )
        row[3] += float(
            (target_energy[local] * weight[:, None, None, None]).sum(dtype=torch.float64)
        )
        row[4] += float(weight.sum()) * pas_cos.shape[1] * pas_cos.shape[2]
        row[5] += float(weight.sum()) * pdp_cos.shape[1] * pdp_cos.shape[2]

    add(accumulator[index], torch.arange(len(prediction), device=prediction.device))
    for group in np.unique(labels):
        local = torch.as_tensor(np.flatnonzero(labels == group), device=prediction.device)
        add(group_accumulator[index, int(group)], local)


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
    phase9_target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    from phase8_anchor_retained_pas_screen import horizontal_shifts

    shifts = horizontal_shifts(pos)
    fold_diagnostics = []
    for fold, row in enumerate(folds):
        val = row["val"]
        labels = row["labels"]
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        external_mask = valid.copy()
        external_mask[val] = False
        external_train = np.flatnonzero(external_mask)
        row["pas_residual"] = external_residual(
            pos,
            val,
            labels,
            anchors,
            row["query_local"],
            phase9_target,
            external_train,
            shifts,
            "none",
        )
        train_mask = external_mask.copy()
        train_mask[val[anchors]] = True
        train = np.flatnonzero(train_mask)
        row["calibrated_weights"], diagnostic = calibrated_weights(
            pos,
            row["query"],
            labels[row["query_local"]],
            train,
            official_counts,
            official_log_d1,
        )
        fold_diagnostics.append(diagnostic)
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

    fold_rows = []
    all_group_accumulators = []
    for fold, row in enumerate(folds):
        labels = row["labels"][row["query_local"]]
        robust_alpha = np.clip(0.75 * gate_alpha[fold], 0.0, 0.6)[:, None, None, None]
        corrected = normalize(row["base"] * np.exp(0.15 * row["pas_residual"]))
        robust_target = normalize(
            (1.0 - robust_alpha) * corrected + robust_alpha * row["local"]
        ).astype(np.float32)
        aggregate_target_all = np.load(
            ROOT / f"matched_phase7_aggregate_portfolio_mean_pas_band24_fold{fold}.npy",
            mmap_mode="r",
        )
        aggregate_target = np.asarray(aggregate_target_all[row["query_local"]])
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        accumulator = np.zeros((1 + len(CANDIDATES), 6), dtype=np.float64)
        group_accumulator = np.zeros((1 + len(CANDIDATES), 11, 6), dtype=np.float64)
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
            aggregate_band = torch.as_tensor(aggregate_target[start:stop].copy(), device=DEVICE)
            robust_band = torch.as_tensor(robust_target[start:stop].copy(), device=DEVICE)
            batch_labels = labels[start:stop]
            update_weighted_scores(
                accumulator,
                group_accumulator,
                0,
                p,
                t,
                truth_pas,
                truth_pdp,
                weights,
                batch_labels,
            )
            for index, (_, agg_scale, residual_scale, agg_groups, residual_groups, iterations) in enumerate(CANDIDATES, 1):
                agg_mask = torch.as_tensor(
                    np.isin(batch_labels, agg_groups).astype(np.float32)[:, None, None, None],
                    device=DEVICE,
                )
                residual_mask = torch.as_tensor(
                    np.isin(batch_labels, residual_groups).astype(np.float32)[:, None, None, None],
                    device=DEVICE,
                )
                desired = normalized(
                    base_band
                    + agg_scale * agg_mask * (aggregate_band - base_band)
                    + residual_scale * residual_mask * (robust_band - base_band)
                )
                value = project(p, base_band, desired, 24, iterations)
                update_weighted_scores(
                    accumulator,
                    group_accumulator,
                    index,
                    value,
                    t,
                    truth_pas,
                    truth_pdp,
                    weights,
                    batch_labels,
                )
        baseline = components(accumulator[0])
        rows = []
        for index, candidate in enumerate(CANDIDATES, 1):
            value = components(accumulator[index])
            rows.append(
                {
                    "name": candidate[0],
                    "aggregate_scale": candidate[1],
                    "residual_scale": candidate[2],
                    "aggregate_groups": candidate[3],
                    "residual_groups": candidate[4],
                    "iterations": candidate[5],
                    **value,
                    "delta_vs_phase6": value["score"] - baseline["score"],
                }
            )
        fold_rows.append({"fold": fold, "baseline": baseline, "rows": rows})
        all_group_accumulators.append(group_accumulator)
        print(
            json.dumps({"stage": "fold", "fold": fold, "best": max(rows, key=lambda x: x["score"])}),
            flush=True,
        )

    composite_baseline_stats = np.zeros(6, dtype=np.float64)
    for group, fold in COMPOSITE_FOLD_BY_GROUP.items():
        composite_baseline_stats += all_group_accumulators[fold][0, group]
    composite_baseline = components(composite_baseline_stats)
    summary = []
    for index, candidate in enumerate(CANDIDATES, 1):
        deltas = np.asarray(
            [fold["rows"][index - 1]["delta_vs_phase6"] for fold in fold_rows]
        )
        composite_stats = np.zeros(6, dtype=np.float64)
        for group, fold in COMPOSITE_FOLD_BY_GROUP.items():
            composite_stats += all_group_accumulators[fold][index, group]
        composite = components(composite_stats)
        summary.append(
            {
                "name": candidate[0],
                "aggregate_scale": candidate[1],
                "residual_scale": candidate[2],
                "aggregate_groups": candidate[3],
                "residual_groups": candidate[4],
                "iterations": candidate[5],
                "fold_deltas": deltas.tolist(),
                "geometry_weighted_delta": float(np.dot(FOLD_WEIGHTS, deltas)),
                "stress_min_delta": float(np.min(deltas[3:])),
                "all_fold_min_delta": float(np.min(deltas)),
                "composite_score": composite["score"],
                "composite_delta": float(composite["score"] - composite_baseline["score"]),
                "robust_proxy": float(
                    min(np.dot(FOLD_WEIGHTS, deltas), composite["score"] - composite_baseline["score"])
                ),
            }
        )
    summary.sort(
        key=lambda item: (
            item["robust_proxy"],
            item["stress_min_delta"],
            item["geometry_weighted_delta"],
        ),
        reverse=True,
    )
    output = {
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "fold_weights": FOLD_WEIGHTS.tolist(),
        "composite_fold_by_group": COMPOSITE_FOLD_BY_GROUP,
        "official_log_d1": official_log_d1,
        "fold_diagnostics": fold_diagnostics,
        "composite_baseline": composite_baseline,
        "folds": fold_rows,
        "summary": summary,
    }
    (ROOT / "phase10_calibrated_pas_residual_joint_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "summary", "top": summary[:10]}), flush=True)


if __name__ == "__main__":
    run()
