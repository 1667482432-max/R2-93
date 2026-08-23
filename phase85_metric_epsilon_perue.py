from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
import phase82_metric_epsilon_calibration as p82


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
PREDECLARE = ROOT / "phase85_metric_epsilon_perue_predeclared.json"
OUTPUT = ROOT / "phase85_metric_epsilon_perue.json"
FROZEN_PREDECLARE_SHA256 = "b3f7bbf8811b5f1c302834163c6b523abacf48e4c582d558eae23cf8f6d69b24"

BETA_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
ALPHA_BOUNDS = (0.25, 12.0)
SCALE_BOUNDS = (0.25, 16.0)
INNER_DEPARTURE_MARGIN = 0.0002
EPS_COS = 1e-30
EPS_NMSE = 1e-20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def batch_statistics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    # The UE dimension is retained; only BS antenna and subcarrier are reduced.
    pred_energy = torch.sum(
        torch.abs(prediction) ** 2, dim=(1, 3), dtype=torch.float64
    )
    cross = torch.sum(
        torch.real(torch.conj(prediction) * target), dim=(1, 3), dtype=torch.float64
    )
    target_energy = torch.sum(
        torch.abs(target) ** 2, dim=(1, 3), dtype=torch.float64
    )

    pred_pas = torch.abs(rp.bs_fft_torch(prediction)) ** 2
    target_pas = torch.abs(rp.bs_fft_torch(target)) ** 2
    pas_num = torch.sum(pred_pas * target_pas, dim=1)  # [N, UE, S]
    pas_pnorm = torch.linalg.vector_norm(pred_pas, dim=1)
    pas_tnorm = torch.linalg.vector_norm(target_pas, dim=1)

    pred_pdp = torch.abs(torch.fft.fft(prediction, dim=-1, norm="ortho")) ** 2
    target_pdp = torch.abs(torch.fft.fft(target, dim=-1, norm="ortho")) ** 2
    pdp_num = torch.sum(pred_pdp * target_pdp, dim=-1)  # [N, BS, UE]
    pdp_pnorm = torch.linalg.vector_norm(pred_pdp, dim=-1)
    pdp_tnorm = torch.linalg.vector_norm(target_pdp, dim=-1)

    return {
        "A": pred_energy.cpu().numpy(),
        "B": cross.cpu().numpy(),
        "T": target_energy.cpu().numpy(),
        "pas_num": _to_numpy(pas_num),
        "pas_den": _to_numpy(pas_pnorm * pas_tnorm),
        "pas_pnorm": _to_numpy(pas_pnorm),
        "pas_tnorm": _to_numpy(pas_tnorm),
        "pdp_num": _to_numpy(pdp_num),
        "pdp_den": _to_numpy(pdp_pnorm * pdp_tnorm),
        "pdp_pnorm": _to_numpy(pdp_pnorm),
        "pdp_tnorm": _to_numpy(pdp_tnorm),
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
    kind: str,
) -> float:
    if kind == "pas":
        scale2 = np.square(scales, dtype=np.float64)[:, :, None]
        weight_view = weights[:, None, None]
    elif kind == "pdp":
        scale2 = np.square(scales, dtype=np.float64)[:, None, :]
        weight_view = weights[:, None, None]
    else:
        raise ValueError(kind)
    similarity = (scale2 * numerator.astype(np.float64)) / np.maximum(
        scale2 * denominator.astype(np.float64), EPS_COS
    )
    both_zero = (scale2 * prediction_norm.astype(np.float64) <= EPS_COS) & (
        target_norm.astype(np.float64) <= EPS_COS
    )
    if np.any(both_zero):
        similarity[both_zero] = 1.0
    return float(
        np.sum(weight_view * similarity, dtype=np.float64)
        / (np.sum(weights, dtype=np.float64) * similarity.shape[1] * similarity.shape[2])
    )


def score_statistics(stats: dict[str, np.ndarray], scales: np.ndarray) -> dict[str, float]:
    weights = stats["weights"].astype(np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != stats["A"].shape:
        raise ValueError(f"scale shape {scales.shape} != A shape {stats['A'].shape}")
    c1 = spectral_component(
        stats["pas_num"], stats["pas_den"], stats["pas_pnorm"], stats["pas_tnorm"],
        weights, scales, "pas",
    )
    c2 = spectral_component(
        stats["pdp_num"], stats["pdp_den"], stats["pdp_pnorm"], stats["pdp_tnorm"],
        weights, scales, "pdp",
    )
    error = stats["A"] * scales**2 - 2.0 * stats["B"] * scales + stats["T"]
    total_error = float(np.sum(weights[:, None] * error, dtype=np.float64))
    total_target = float(np.sum(weights[:, None] * stats["T"], dtype=np.float64))
    c3 = total_error / max(total_target, EPS_NMSE)
    return {
        "c1_pas": c1,
        "c2_pdp": c2,
        "c3_nmse": float(c3),
        "score": float(0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3)),
    }


def fit_calibrator(
    folds: list[dict[str, np.ndarray]], indices: list[int], beta: float
) -> dict[str, float]:
    energy = np.concatenate([folds[index]["A"].reshape(-1) for index in indices])
    cross = np.concatenate([folds[index]["B"].reshape(-1) for index in indices])
    row_weights = np.concatenate([
        np.repeat(folds[index]["weights"], 4) for index in indices
    ]).astype(np.float64)
    log_energy = np.log(np.maximum(energy, 1e-30))
    q05 = p82.weighted_quantile(log_energy, row_weights, 0.05)
    q50 = p82.weighted_quantile(log_energy, row_weights, 0.50)
    q95 = p82.weighted_quantile(log_energy, row_weights, 0.95)
    width = max(q95 - q05, 1e-12)
    u = 2.0 * (np.clip(log_energy, q05, q95) - q05) / width - 1.0
    basis = np.exp(beta * u)
    numerator = float(np.sum(row_weights * cross * basis, dtype=np.float64))
    denominator = float(
        np.sum(row_weights * energy * np.square(basis), dtype=np.float64)
    )
    alpha_raw = numerator / max(denominator, 1e-300)
    return {
        "beta": float(beta),
        "q05_log_A": q05,
        "q50_log_A": q50,
        "q95_log_A": q95,
        "alpha_raw": float(alpha_raw),
        "alpha": float(np.clip(alpha_raw, *ALPHA_BOUNDS)),
    }


def apply_calibrator(stats: dict[str, np.ndarray], model: dict[str, float]) -> np.ndarray:
    log_energy = np.log(np.maximum(stats["A"], 1e-30))
    q05 = model["q05_log_A"]
    width = max(model["q95_log_A"] - q05, 1e-12)
    u = 2.0 * (np.clip(log_energy, q05, model["q95_log_A"]) - q05) / width - 1.0
    return np.clip(model["alpha"] * np.exp(model["beta"] * u), *SCALE_BOUNDS)


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
            scales = apply_calibrator(folds[inner_val], model)
            value = score_statistics(folds[inner_val], scales)
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
    difference = np.asarray(best["inner_scores"]) - np.asarray(beta0["inner_scores"])
    departure = bool(
        float(np.mean(difference)) >= INNER_DEPARTURE_MARGIN
        and int(np.sum(difference >= 0.0)) >= 3
    )
    selected = float(best["beta"] if departure else 0.0)
    return selected, {
        "grid": records,
        "best_nonzero_beta": float(best["beta"]),
        "best_nonzero_minus_beta0_by_inner_fold": difference.tolist(),
        "best_nonzero_minus_beta0_mean": float(np.mean(difference)),
        "best_nonzero_no_worse_count": int(np.sum(difference >= 0.0)),
        "departure_gate_passed": departure,
        "selected_beta": selected,
    }


def summarize_scales(scales: np.ndarray) -> dict[str, float]:
    flat = scales.reshape(-1)
    return {
        "minimum": float(np.min(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "median": float(np.median(flat)),
        "q95": float(np.quantile(flat, 0.95)),
        "maximum": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }


@torch.no_grad()
def reconstruct_phase9_statistics() -> tuple[list[dict[str, np.ndarray]], dict, dict]:
    folds, pos, channel, energy, official_counts, actual_counts = p82.prepare_folds()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    _, actual_fraction, _ = p82.official_geometry(pos, energy, test_pos)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    shifts = p82.horizontal_shifts(pos)
    valid = energy > 0
    output: list[dict[str, np.ndarray]] = []

    for fold, row in enumerate(folds):
        anchors = p82.mapped_anchors(
            pos, row["val"], row["labels"], actual_fraction, official_counts
        )
        train_mask = valid.copy()
        train_mask[row["val"]] = False
        external_train = np.flatnonzero(train_mask)
        pas_residual = p82.external_residual(
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
        pdp_residual = p82.external_pdp_residual(
            pos, channel, row["val"], row["labels"], anchors, row["query_local"],
            external_train, truth_pdp_all,
        )
        train_x = np.concatenate([folds[i]["x"] for i in range(5) if i != fold])
        train_y = np.concatenate([folds[i]["gain_grid"] for i in range(5) if i != fold])
        train_w = np.concatenate([folds[i]["weights"] for i in range(5) if i != fold])
        gate = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=80, max_features=0.7,
            n_jobs=-1, random_state=52180,
        )
        gate.fit(train_x, train_y, sample_weight=train_w)
        gate_alpha = p82.ALPHA_GRID[np.argmax(gate.predict(row["x"]), axis=1)]
        alpha = np.clip(p82.LOCAL_GATE_SCALE * gate_alpha, 0.0, 0.6)[:, None, None, None]
        corrected = p82.normalize(row["base"] * np.exp(p82.PAS_RESIDUAL_ALPHA * pas_residual))
        desired_pas = p82.normalize(
            (1.0 - alpha) * corrected + alpha * row["local"]
        ).astype(np.float32)
        desired_pdp = p82.normalize_last(
            base_pdp * np.exp(p82.PDP_RESIDUAL_ALPHA * pdp_residual)
        ).astype(np.float32)

        phase6 = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        batches: list[dict[str, np.ndarray]] = []
        finite = True
        for start in range(0, len(row["query"]), 2):
            stop = min(start + 2, len(row["query"]))
            local_query = row["query_local"][start:stop]
            prediction = torch.as_tensor(
                np.asarray(phase6[local_query]).copy(), device=DEVICE
            )
            truth = torch.as_tensor(
                np.asarray(channel[row["query"][start:stop]]).copy(), device=DEVICE
            )
            phase9 = p82.project_joint(
                prediction,
                torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE),
                torch.as_tensor(desired_pas[start:stop], device=DEVICE),
                torch.as_tensor(desired_pdp[start:stop], device=DEVICE),
                p82.PROJECTION_ITERATIONS,
            )
            finite = finite and bool(torch.isfinite(phase9).all())
            batches.append(batch_statistics(phase9, truth))
        stats = concatenate_batches(batches)
        stats["weights"] = row["weights"].astype(np.float32).astype(np.float64)
        output.append(stats)
        base = score_statistics(stats, np.ones_like(stats["A"]))
        expected = p82.expected_phase9_fold(fold)
        print(json.dumps({
            "stage": "phase9_perue_statistics", "fold": fold,
            "rows": len(row["query"]), "all_finite": finite,
            "base": base,
            "reconstruction_error": {
                key: float(base[key] - expected[key])
                for key in ("c1_pas", "c2_pdp", "c3_nmse", "score")
            },
        }), flush=True)
        del pas_residual, base_pdp, pdp_residual, desired_pas, desired_pdp, batches
        gc.collect()
        torch.cuda.empty_cache()
    return output, official_counts, actual_counts


def run() -> None:
    if sha256(PREDECLARE) != FROZEN_PREDECLARE_SHA256:
        raise RuntimeError("Phase85 predeclaration hash mismatch")
    predeclare = json.loads(PREDECLARE.read_text(encoding="utf-8"))
    if tuple(float(value) for value in predeclare["candidate"]["fixed_beta_grid"]) != BETA_GRID:
        raise RuntimeError("beta grid differs from Phase85 predeclaration")
    if sha256(ROOT / "metrics.py") != predeclare["immutable_inputs"]["metrics_py_sha256"]:
        raise RuntimeError("metrics.py differs from Phase85 predeclaration")

    folds, official_counts, actual_counts = reconstruct_phase9_statistics()
    baseline = [score_statistics(row, np.ones_like(row["A"])) for row in folds]
    reconstruction_errors = []
    for fold in range(5):
        expected = p82.expected_phase9_fold(fold)
        reconstruction_errors.append({
            key: float(baseline[fold][key] - expected[key])
            for key in ("c1_pas", "c2_pdp", "c3_nmse", "score")
        })
    max_reconstruction_error = max(
        abs(value) for row in reconstruction_errors for value in row.values()
    )

    outer_rows: list[dict[str, Any]] = []
    for heldout in range(5):
        source = [fold for fold in range(5) if fold != heldout]
        beta, selection = select_beta_nested(folds, source, baseline)
        model = fit_calibrator(folds, source, beta)
        scales = apply_calibrator(folds[heldout], model)
        candidate = score_statistics(folds[heldout], scales)
        fixed5 = score_statistics(folds[heldout], np.full_like(scales, 5.0))
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
            "blind_whole_row_fixed5": {
                "score": fixed5,
                "score_delta": float(fixed5["score"] - baseline[heldout]["score"]),
            },
        }
        outer_rows.append(row)
        print(json.dumps({
            "stage": "outer_fold", "heldout_fold": heldout,
            "selected_beta": beta, "alpha": model["alpha"],
            "scale_summary": row["scale_summary"],
            "score_delta": row["score_delta"],
        }), flush=True)

    deltas = np.asarray([row["score_delta"] for row in outer_rows], dtype=np.float64)
    mean = float(np.mean(deltas))
    lcb = float(mean - np.std(deltas, ddof=0))
    gates = {
        "mean_at_least_0.00075": mean >= 0.00075,
        "at_least_four_of_five_positive": int(np.sum(deltas > 0.0)) >= 4,
        "lcb_strictly_positive": lcb > 0.0,
        "minimum_not_below_minus_0.0003": float(np.min(deltas)) >= -0.0003,
        "all_values_finite": bool(np.all(np.isfinite(deltas))),
        "phase9_reconstruction_within_0.000001": max_reconstruction_error <= 1e-6,
    }
    payload = {
        "predeclare": PREDECLARE.name,
        "predeclare_sha256": sha256(PREDECLARE),
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "configuration_modified_after_predeclaration": False,
        "phase82_or_phase81_results_used_to_modify_rule": False,
        "test_channel_generated": False,
        "test_artifact_built": False,
        "uploaded": False,
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "zero_channel_outliers_excluded": int(np.sum(np.load(ROOT / "train_energy.npy") <= 0)),
        "phase9_reconstruction_errors": reconstruction_errors,
        "phase9_reconstruction_max_abs_error": float(max_reconstruction_error),
        "outer_folds": outer_rows,
        "summary": {
            "score_deltas": deltas.tolist(),
            "mean_score_delta": mean,
            "population_std_score_delta": float(np.std(deltas, ddof=0)),
            "lcb_mean_minus_population_std": lcb,
            "minimum_score_delta": float(np.min(deltas)),
            "positive_folds": int(np.sum(deltas > 0.0)),
            "selected_betas": [row["final_model"]["beta"] for row in outer_rows],
            "selected_alphas": [row["final_model"]["alpha"] for row in outer_rows],
            "maximum_pas_delta_abs": float(max(
                abs(row["component_deltas"]["c1_pas"]) for row in outer_rows
            )),
            "maximum_pdp_delta_abs": float(max(
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
