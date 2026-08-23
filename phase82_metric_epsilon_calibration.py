from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
from phase8_anchor_local_gate_channel_validation import ALPHA_GRID, prepare_folds
from phase8_anchor_retained_pas_screen import horizontal_shifts, mapped_anchors, normalize
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_anchor_joint_channel_validation import normalize_last, project_joint
from phase9_buildable_full_pdp_screen import descriptor as pdp_descriptor
from phase9_buildable_residual_gate_pas_screen import external_residual


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
PREDECLARE = ROOT / "phase82_metric_epsilon_calibration_predeclared.json"
OUTPUT = ROOT / "phase82_metric_epsilon_calibration.json"

PAS_RESIDUAL_ALPHA = 0.15
LOCAL_GATE_SCALE = 0.75
PDP_RESIDUAL_ALPHA = 0.025
PROJECTION_ITERATIONS = 12
BETA_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
ALPHA_BOUNDS = (0.25, 12.0)
SCALE_BOUNDS = (0.25, 16.0)
INNER_DEPARTURE_MARGIN = 0.00015
EPS_COS = 1e-30
EPS_NMSE = 1e-20
FIXED_AUDIT_SCALES = (0.25, 1.0, 5.0, 16.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_pdp_residual(
    pos: np.ndarray,
    channel: np.ndarray,
    val: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    query_local: np.ndarray,
    external_train: np.ndarray,
    truth_all: np.ndarray,
) -> np.ndarray:
    distance_external, local_external = cKDTree(pos[external_train, :2]).query(
        pos[val[anchors], :2], k=8
    )
    external_indices = external_train[local_external]
    unique_indices, inverse = np.unique(external_indices, return_inverse=True)
    external_pdp = pdp_descriptor(channel, unique_indices)[inverse].reshape(
        len(anchors), 8, 256, 4, 192
    )
    scale = np.maximum(np.median(distance_external, axis=1, keepdims=True), 1.0)
    weight = 1.0 / np.maximum(distance_external + 0.10 * scale, 0.25) ** 3.0
    weight /= weight.sum(1, keepdims=True)
    anchor_base = np.einsum("nk,nkaus->naus", weight, external_pdp, optimize=True)
    anchor_truth = np.asarray(truth_all[anchors])
    epsilon = 1e-4 / 192
    anchor_log = np.clip(
        np.log((anchor_truth + epsilon) / (anchor_base + epsilon)), -2.0, 2.0
    )
    output = np.zeros((len(query_local), 256, 4, 192), dtype=np.float32)
    for group in np.unique(labels[query_local]):
        rows = np.flatnonzero(labels[query_local] == group)
        group_anchor_local = np.flatnonzero(labels[anchors] == group)
        if len(group_anchor_local) == 0:
            continue
        k = min(4, len(group_anchor_local))
        distance, local = cKDTree(pos[val[anchors[group_anchor_local]], :2]).query(
            pos[val[query_local[rows]], :2], k=k
        )
        distance = np.asarray(distance)
        local = np.asarray(local)
        if k == 1:
            distance = distance[:, None]
            local = local[:, None]
        local_scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
        local_weight = 1.0 / np.maximum(distance + 0.10 * local_scale, 0.25) ** 0.5
        local_weight /= local_weight.sum(1, keepdims=True)
        output[rows] = np.einsum(
            "rk,rkaus->raus",
            local_weight,
            anchor_log[group_anchor_local[local]],
            optimize=True,
        )
    return output


def expected_phase9_fold(fold: int) -> dict[str, float]:
    payload = json.loads(
        (ROOT / "phase9_buildable_joint_channel_validation.json").read_text(
            encoding="utf-8"
        )
    )
    fold_payload = next(item for item in payload["folds"] if int(item["fold"]) == fold)
    return next(
        item
        for item in fold_payload["rows"]
        if item["pas_label"] == "robust"
        and float(item["pdp_alpha"]) == PDP_RESIDUAL_ALPHA
    )


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def batch_sufficient_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, np.ndarray]:
    pred_energy = torch.sum(torch.abs(prediction) ** 2, dim=(1, 2, 3), dtype=torch.float64)
    cross = torch.sum(
        torch.real(torch.conj(prediction) * target), dim=(1, 2, 3), dtype=torch.float64
    )
    target_energy = torch.sum(torch.abs(target) ** 2, dim=(1, 2, 3), dtype=torch.float64)

    pred_pas = torch.abs(rp.bs_fft_torch(prediction)) ** 2
    target_pas = torch.abs(rp.bs_fft_torch(target)) ** 2
    pas_num = torch.sum(pred_pas * target_pas, dim=1)
    pas_pnorm = torch.linalg.vector_norm(pred_pas, dim=1)
    pas_tnorm = torch.linalg.vector_norm(target_pas, dim=1)
    pas_den = pas_pnorm * pas_tnorm

    pred_pdp = torch.abs(torch.fft.fft(prediction, dim=-1, norm="ortho")) ** 2
    target_pdp = torch.abs(torch.fft.fft(target, dim=-1, norm="ortho")) ** 2
    pdp_num = torch.sum(pred_pdp * target_pdp, dim=-1)
    pdp_pnorm = torch.linalg.vector_norm(pred_pdp, dim=-1)
    pdp_tnorm = torch.linalg.vector_norm(target_pdp, dim=-1)
    pdp_den = pdp_pnorm * pdp_tnorm

    return {
        "A": pred_energy.cpu().numpy(),
        "B": cross.cpu().numpy(),
        "T": target_energy.cpu().numpy(),
        "pas_num": _to_numpy(pas_num).reshape(len(prediction), -1),
        "pas_den": _to_numpy(pas_den).reshape(len(prediction), -1),
        "pas_pnorm": _to_numpy(pas_pnorm).reshape(len(prediction), -1),
        "pas_tnorm": _to_numpy(pas_tnorm).reshape(len(prediction), -1),
        "pdp_num": _to_numpy(pdp_num).reshape(len(prediction), -1),
        "pdp_den": _to_numpy(pdp_den).reshape(len(prediction), -1),
        "pdp_pnorm": _to_numpy(pdp_pnorm).reshape(len(prediction), -1),
        "pdp_tnorm": _to_numpy(pdp_tnorm).reshape(len(prediction), -1),
    }


def concatenate_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in batches[0]}


def spectral_component(
    numerator: np.ndarray,
    denominator: np.ndarray,
    prediction_norm: np.ndarray,
    target_norm: np.ndarray,
    weights: np.ndarray,
    scales: np.ndarray,
) -> float:
    scale2 = np.square(scales, dtype=np.float64)[:, None]
    similarity = (scale2 * numerator.astype(np.float64)) / np.maximum(
        scale2 * denominator.astype(np.float64), EPS_COS
    )
    both_zero = (scale2 * prediction_norm.astype(np.float64) <= EPS_COS) & (
        target_norm.astype(np.float64) <= EPS_COS
    )
    if np.any(both_zero):
        similarity[both_zero] = 1.0
    return float(
        np.sum(weights[:, None] * similarity, dtype=np.float64)
        / (np.sum(weights, dtype=np.float64) * similarity.shape[1])
    )


def score_statistics(stats: dict[str, np.ndarray], scales: np.ndarray) -> dict[str, float]:
    weights = stats["weights"].astype(np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    c1 = spectral_component(
        stats["pas_num"], stats["pas_den"], stats["pas_pnorm"], stats["pas_tnorm"],
        weights, scales,
    )
    c2 = spectral_component(
        stats["pdp_num"], stats["pdp_den"], stats["pdp_pnorm"], stats["pdp_tnorm"],
        weights, scales,
    )
    error = stats["A"] * scales**2 - 2.0 * stats["B"] * scales + stats["T"]
    total_error = float(np.sum(weights * error, dtype=np.float64))
    total_target = float(np.sum(weights * stats["T"], dtype=np.float64))
    c3 = total_error / max(total_target, EPS_NMSE)
    score = 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3)
    return {
        "c1_pas": c1,
        "c2_pdp": c2,
        "c3_nmse": float(c3),
        "score": float(score),
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_value = values[order]
    sorted_weight = weights[order]
    cumulative = np.cumsum(sorted_weight, dtype=np.float64)
    if cumulative[-1] <= 0:
        raise RuntimeError("non-positive total calibration weight")
    return float(np.interp(q * cumulative[-1], cumulative, sorted_value))


def concatenate_rows(folds: list[dict[str, np.ndarray]], indices: list[int]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([folds[index][key] for index in indices], axis=0)
        for key in ("A", "B", "T", "weights")
    }


def fit_calibrator(
    folds: list[dict[str, np.ndarray]], indices: list[int], beta: float
) -> dict[str, float]:
    rows = concatenate_rows(folds, indices)
    log_energy = np.log(np.maximum(rows["A"], 1e-30))
    weights = rows["weights"].astype(np.float64)
    q05 = weighted_quantile(log_energy, weights, 0.05)
    q50 = weighted_quantile(log_energy, weights, 0.50)
    q95 = weighted_quantile(log_energy, weights, 0.95)
    width = max(q95 - q05, 1e-12)
    u = 2.0 * (np.clip(log_energy, q05, q95) - q05) / width - 1.0
    basis = np.exp(beta * u)
    numerator = float(np.sum(weights * rows["B"] * basis, dtype=np.float64))
    denominator = float(
        np.sum(weights * rows["A"] * np.square(basis), dtype=np.float64)
    )
    alpha_raw = numerator / max(denominator, 1e-300)
    alpha = float(np.clip(alpha_raw, *ALPHA_BOUNDS))
    return {
        "beta": float(beta),
        "q05_log_A": q05,
        "q50_log_A": q50,
        "q95_log_A": q95,
        "alpha_raw": float(alpha_raw),
        "alpha": alpha,
    }


def apply_calibrator(stats: dict[str, np.ndarray], model: dict[str, float]) -> np.ndarray:
    log_energy = np.log(np.maximum(stats["A"], 1e-30))
    q05 = model["q05_log_A"]
    width = max(model["q95_log_A"] - q05, 1e-12)
    u = 2.0 * (np.clip(log_energy, q05, model["q95_log_A"]) - q05) / width - 1.0
    scale = model["alpha"] * np.exp(model["beta"] * u)
    return np.clip(scale, *SCALE_BOUNDS)


def select_beta_nested(
    folds: list[dict[str, np.ndarray]], source: list[int], baseline: list[dict[str, float]]
) -> tuple[float, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for beta in BETA_GRID:
        scores = []
        deltas = []
        fits = []
        for inner_val in source:
            inner_train = [index for index in source if index != inner_val]
            model = fit_calibrator(folds, inner_train, beta)
            value = score_statistics(folds[inner_val], apply_calibrator(folds[inner_val], model))
            scores.append(value["score"])
            deltas.append(value["score"] - baseline[inner_val]["score"])
            fits.append({
                "inner_validation_fold": inner_val,
                "inner_training_folds": inner_train,
                "model": model,
                "score": value["score"],
                "delta_vs_identity": deltas[-1],
            })
        records.append({
            "beta": beta,
            "inner_scores": scores,
            "inner_deltas_vs_identity": deltas,
            "mean_inner_score": float(np.mean(scores)),
            "mean_inner_delta_vs_identity": float(np.mean(deltas)),
            "fits": fits,
        })

    beta0 = next(row for row in records if row["beta"] == 0.0)
    nonzero = sorted(
        [row for row in records if row["beta"] != 0.0],
        key=lambda row: (-row["mean_inner_score"], abs(row["beta"]), row["beta"]),
    )
    best = nonzero[0]
    pairwise = np.asarray(best["inner_scores"]) - np.asarray(beta0["inner_scores"])
    departure = bool(
        float(np.mean(pairwise)) >= INNER_DEPARTURE_MARGIN
        and int(np.sum(pairwise >= 0.0)) >= 3
    )
    selected = float(best["beta"] if departure else 0.0)
    audit = {
        "grid": records,
        "best_nonzero_beta": float(best["beta"]),
        "best_nonzero_minus_beta0_by_inner_fold": pairwise.tolist(),
        "best_nonzero_minus_beta0_mean": float(np.mean(pairwise)),
        "best_nonzero_no_worse_count": int(np.sum(pairwise >= 0.0)),
        "departure_gate_passed": departure,
        "selected_beta": selected,
    }
    return selected, audit


def clamp_counts(stats: dict[str, np.ndarray], scales: np.ndarray, prefix: str) -> dict[str, Any]:
    weights = stats["weights"].astype(np.float64)
    scale2 = np.square(scales, dtype=np.float64)[:, None]
    denominator = scale2 * stats[f"{prefix}_den"].astype(np.float64)
    active = denominator <= EPS_COS
    both_zero = (
        scale2 * stats[f"{prefix}_pnorm"].astype(np.float64) <= EPS_COS
    ) & (stats[f"{prefix}_tnorm"].astype(np.float64) <= EPS_COS)
    total_weighted_units = float(np.sum(weights) * active.shape[1])
    return {
        "raw_active_units": int(np.sum(active)),
        "raw_total_units": int(active.size),
        "raw_active_fraction": float(np.mean(active)),
        "weighted_active_fraction": float(
            np.sum(weights[:, None] * active, dtype=np.float64) / total_weighted_units
        ),
        "raw_both_zero_overrides": int(np.sum(both_zero)),
        "weighted_both_zero_fraction": float(
            np.sum(weights[:, None] * both_zero, dtype=np.float64) / total_weighted_units
        ),
    }


def summarize_scales(scales: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(scales)),
        "q05": float(np.quantile(scales, 0.05)),
        "median": float(np.median(scales)),
        "q95": float(np.quantile(scales, 0.95)),
        "maximum": float(np.max(scales)),
        "mean": float(np.mean(scales)),
    }


@torch.no_grad()
def reconstruct_phase9_statistics() -> tuple[list[dict[str, np.ndarray]], dict, dict]:
    folds, pos, channel, energy, official_counts, actual_counts = prepare_folds()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    shifts = horizontal_shifts(pos)
    valid = energy > 0

    output: list[dict[str, np.ndarray]] = []
    for fold, row in enumerate(folds):
        anchors = mapped_anchors(
            pos, row["val"], row["labels"], actual_fraction, official_counts
        )
        train_mask = valid.copy()
        train_mask[row["val"]] = False
        external_train = np.flatnonzero(train_mask)
        pas_residual = external_residual(
            pos, row["val"], row["labels"], anchors, row["query_local"],
            target, external_train, shifts, "none",
        )
        base_pdp_all = np.load(
            ROOT / f"phase9_anchor_fold{fold}_base_pdp_band192.npy", mmap_mode="r"
        )
        truth_pdp_all = np.load(
            ROOT / f"phase9_anchor_fold{fold}_truth_pdp_band192.npy", mmap_mode="r"
        )
        base_pdp = np.asarray(base_pdp_all[row["query_local"]])
        pdp_residual = external_pdp_residual(
            pos, channel, row["val"], row["labels"], anchors, row["query_local"],
            external_train, truth_pdp_all,
        )

        train_x = np.concatenate([folds[i]["x"] for i in range(5) if i != fold])
        train_y = np.concatenate([folds[i]["gain_grid"] for i in range(5) if i != fold])
        train_w = np.concatenate([folds[i]["weights"] for i in range(5) if i != fold])
        gate = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=80,
            max_features=0.7,
            n_jobs=-1,
            random_state=52180,
        )
        gate.fit(train_x, train_y, sample_weight=train_w)
        gate_alpha = ALPHA_GRID[np.argmax(gate.predict(row["x"]), axis=1)]
        alpha = np.clip(LOCAL_GATE_SCALE * gate_alpha, 0.0, 0.6)[:, None, None, None]
        corrected = normalize(row["base"] * np.exp(PAS_RESIDUAL_ALPHA * pas_residual))
        desired_pas = normalize((1.0 - alpha) * corrected + alpha * row["local"]).astype(
            np.float32
        )
        desired_pdp = normalize_last(
            base_pdp * np.exp(PDP_RESIDUAL_ALPHA * pdp_residual)
        ).astype(np.float32)

        query_local = row["query_local"]
        query = row["query"]
        phase6 = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        batches: list[dict[str, np.ndarray]] = []
        all_finite = True
        for start in range(0, len(query), 2):
            stop = min(start + 2, len(query))
            local_query = query_local[start:stop]
            prediction = torch.as_tensor(
                np.asarray(phase6[local_query]).copy(), device=DEVICE
            )
            truth = torch.as_tensor(
                np.asarray(channel[query[start:stop]]).copy(), device=DEVICE
            )
            phase9 = project_joint(
                prediction,
                torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE),
                torch.as_tensor(desired_pas[start:stop], device=DEVICE),
                torch.as_tensor(desired_pdp[start:stop], device=DEVICE),
                PROJECTION_ITERATIONS,
            )
            all_finite = all_finite and bool(torch.isfinite(phase9).all())
            batches.append(batch_sufficient_statistics(phase9, truth))
        stats = concatenate_batches(batches)
        stats["weights"] = row["weights"].astype(np.float32).astype(np.float64)
        stats["fold"] = np.full(len(query), fold, dtype=np.int64)
        output.append(stats)

        base = score_statistics(stats, np.ones(len(query), dtype=np.float64))
        expected = expected_phase9_fold(fold)
        reconstruction = {
            key: float(base[key] - expected[key])
            for key in ("c1_pas", "c2_pdp", "c3_nmse", "score")
        }
        print(json.dumps({
            "stage": "phase9_sufficient_statistics",
            "fold": fold,
            "rows": len(query),
            "all_finite": all_finite,
            "base": base,
            "reconstruction_error": reconstruction,
        }), flush=True)

        del pas_residual, base_pdp, pdp_residual, desired_pas, desired_pdp, batches
        gc.collect()
        torch.cuda.empty_cache()

    return output, official_counts, actual_counts


def run() -> None:
    predeclare = json.loads(PREDECLARE.read_text(encoding="utf-8"))
    if tuple(float(x) for x in predeclare["candidate"]["fixed_beta_grid"]) != BETA_GRID:
        raise RuntimeError("beta grid differs from predeclaration")
    if sha256(ROOT / "metrics.py") != predeclare["immutable_inputs"]["metrics_py_sha256"]:
        raise RuntimeError("metrics.py differs from frozen predeclaration")
    if sha256(ROOT / "phase9_buildable_joint_channel_validation.py") != predeclare[
        "immutable_inputs"
    ]["phase9_validation_script_sha256"]:
        raise RuntimeError("P9 reference script differs from frozen predeclaration")

    folds, official_counts, actual_counts = reconstruct_phase9_statistics()
    baseline = [score_statistics(row, np.ones(len(row["A"]))) for row in folds]
    expected = [expected_phase9_fold(index) for index in range(5)]
    reconstruction_errors = [
        {
            key: float(baseline[index][key] - expected[index][key])
            for key in ("c1_pas", "c2_pdp", "c3_nmse", "score")
        }
        for index in range(5)
    ]
    max_reconstruction_error = max(
        abs(value) for row in reconstruction_errors for value in row.values()
    )

    outer_rows: list[dict[str, Any]] = []
    for heldout in range(5):
        source = [index for index in range(5) if index != heldout]
        selected_beta, selection = select_beta_nested(folds, source, baseline)
        model = fit_calibrator(folds, source, selected_beta)
        scales = apply_calibrator(folds[heldout], model)
        candidate = score_statistics(folds[heldout], scales)
        fixed5 = score_statistics(folds[heldout], np.full(len(scales), 5.0))
        global_model = fit_calibrator(folds, source, 0.0)
        global_scales = apply_calibrator(folds[heldout], global_model)
        global_candidate = score_statistics(folds[heldout], global_scales)
        row = {
            "heldout_fold": heldout,
            "source_folds": source,
            "selection": selection,
            "final_model": model,
            "scale_summary": summarize_scales(scales),
            "reference_phase9": baseline[heldout],
            "candidate": candidate,
            "score_delta": float(candidate["score"] - baseline[heldout]["score"]),
            "component_deltas": {
                key: float(candidate[key] - baseline[heldout][key])
                for key in ("c1_pas", "c2_pdp", "c3_nmse")
            },
            "analytic_global_beta0": {
                "model": global_model,
                "score": global_candidate,
                "score_delta": float(
                    global_candidate["score"] - baseline[heldout]["score"]
                ),
            },
            "blind_fixed_scale5": {
                "score": fixed5,
                "score_delta": float(fixed5["score"] - baseline[heldout]["score"]),
            },
            "selected_clamp": {
                "pas": clamp_counts(folds[heldout], scales, "pas"),
                "pdp": clamp_counts(folds[heldout], scales, "pdp"),
            },
        }
        outer_rows.append(row)
        print(json.dumps({
            "stage": "outer_fold",
            "heldout_fold": heldout,
            "selected_beta": selected_beta,
            "alpha": model["alpha"],
            "scale_summary": row["scale_summary"],
            "score_delta": row["score_delta"],
            "fixed5_delta": row["blind_fixed_scale5"]["score_delta"],
            "global_beta0_delta": row["analytic_global_beta0"]["score_delta"],
        }), flush=True)

    fixed_clamp_audit: list[dict[str, Any]] = []
    for fold, stats in enumerate(folds):
        for scale in FIXED_AUDIT_SCALES:
            scales = np.full(len(stats["A"]), scale, dtype=np.float64)
            fixed_clamp_audit.append({
                "fold": fold,
                "scale": scale,
                "pas": clamp_counts(stats, scales, "pas"),
                "pdp": clamp_counts(stats, scales, "pdp"),
                "components": score_statistics(stats, scales),
            })

    deltas = np.asarray([row["score_delta"] for row in outer_rows], dtype=np.float64)
    fixed5_deltas = np.asarray(
        [row["blind_fixed_scale5"]["score_delta"] for row in outer_rows], dtype=np.float64
    )
    global_deltas = np.asarray(
        [row["analytic_global_beta0"]["score_delta"] for row in outer_rows],
        dtype=np.float64,
    )
    mean = float(np.mean(deltas))
    lcb = float(mean - np.std(deltas, ddof=0))
    gates = {
        "mean_at_least_0.00075": mean >= 0.00075,
        "at_least_four_of_five_positive": int(np.sum(deltas > 0.0)) >= 4,
        "lcb_strictly_positive": lcb > 0.0,
        "minimum_not_below_minus_0.0003": float(np.min(deltas)) >= -0.0003,
        "all_values_finite": bool(
            np.all(np.isfinite(deltas))
            and all(np.all(np.isfinite(apply_calibrator(
                folds[row["heldout_fold"]], row["final_model"]
            ))) for row in outer_rows)
        ),
        "phase9_reconstruction_within_0.000001": max_reconstruction_error <= 1e-6,
    }
    payload = {
        "predeclare": PREDECLARE.name,
        "predeclare_sha256": sha256(PREDECLARE),
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "configuration_modified_after_predeclaration": False,
        "phase81_results_read_or_used": False,
        "test_channel_generated": False,
        "test_artifact_built": False,
        "uploaded": False,
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "zero_channel_outliers_excluded": int(np.sum(np.load(ROOT / "train_energy.npy") <= 0)),
        "metric_formula_audit": predeclare["metric_audit"],
        "phase9_reconstruction_errors": reconstruction_errors,
        "phase9_reconstruction_max_abs_error": float(max_reconstruction_error),
        "outer_folds": outer_rows,
        "fixed_scale_clamp_audit": fixed_clamp_audit,
        "summary": {
            "score_deltas": deltas.tolist(),
            "mean_score_delta": mean,
            "population_std_score_delta": float(np.std(deltas, ddof=0)),
            "lcb_mean_minus_population_std": lcb,
            "minimum_score_delta": float(np.min(deltas)),
            "positive_folds": int(np.sum(deltas > 0.0)),
            "selected_betas": [row["final_model"]["beta"] for row in outer_rows],
            "selected_alphas": [row["final_model"]["alpha"] for row in outer_rows],
            "blind_fixed5_deltas": fixed5_deltas.tolist(),
            "blind_fixed5_mean_delta": float(np.mean(fixed5_deltas)),
            "analytic_global_beta0_deltas": global_deltas.tolist(),
            "analytic_global_beta0_mean_delta": float(np.mean(global_deltas)),
            "maximum_selected_pas_delta_abs": float(max(
                abs(row["component_deltas"]["c1_pas"]) for row in outer_rows
            )),
            "maximum_selected_pdp_delta_abs": float(max(
                abs(row["component_deltas"]["c2_pdp"]) for row in outer_rows
            )),
            "gates": gates,
            "passed": all(gates.values()),
            "next_step": "report_only_no_test_build_or_upload",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "complete", **payload["summary"]}), flush=True)


if __name__ == "__main__":
    run()
