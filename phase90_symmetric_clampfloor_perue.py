from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase82_metric_epsilon_calibration as p82
import phase85_metric_epsilon_perue as p85


ROOT = Path(__file__).resolve().parent
PREDECLARE = ROOT / "phase90_symmetric_clampfloor_perue_predeclared.json"
OUTPUT = ROOT / "phase90_symmetric_clampfloor_perue.json"
FROZEN_PREDECLARE_SHA256 = "b19d1da428860a20d54cce6f72f43e827219875796255eca22177fa23a68314c"
LOW_TAIL_QUANTILE = 0.01
SYMMETRIC_BOUNDARY = 1e-15
SCALE_BOUNDS = (1.0, 5.0)
EPS_COS = 1e-30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_only_scales(stats: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    q_pas = np.quantile(stats["pas_pnorm"].astype(np.float64), LOW_TAIL_QUANTILE, axis=2)
    q_pdp = np.quantile(stats["pdp_pnorm"].astype(np.float64), LOW_TAIL_QUANTILE, axis=1)
    q = np.minimum(q_pas, q_pdp)
    raw = np.sqrt(SYMMETRIC_BOUNDARY / np.maximum(q, 1e-30))
    scales = np.clip(raw, *SCALE_BOUNDS)
    return scales, {"q_pas": q_pas, "q_pdp": q_pdp, "q": q, "raw": raw}


def clamp_report(
    stats: dict[str, np.ndarray], scales: np.ndarray, kind: str
) -> dict[str, float | int]:
    weights = stats["weights"].astype(np.float64)
    if kind == "pas":
        scale2 = scales[:, :, None] ** 2
    elif kind == "pdp":
        scale2 = scales[:, None, :] ** 2
    else:
        raise ValueError(kind)
    active = scale2 * stats[f"{kind}_den"].astype(np.float64) <= EPS_COS
    both_zero = (
        scale2 * stats[f"{kind}_pnorm"].astype(np.float64) <= EPS_COS
    ) & (stats[f"{kind}_tnorm"].astype(np.float64) <= EPS_COS)
    total_weighted = float(np.sum(weights) * active.shape[1] * active.shape[2])
    return {
        "raw_active_units": int(np.sum(active)),
        "raw_total_units": int(active.size),
        "raw_active_fraction": float(np.mean(active)),
        "weighted_active_fraction": float(
            np.sum(weights[:, None, None] * active, dtype=np.float64) / total_weighted
        ),
        "raw_both_zero_overrides": int(np.sum(both_zero)),
        "weighted_both_zero_fraction": float(
            np.sum(weights[:, None, None] * both_zero, dtype=np.float64) / total_weighted
        ),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1)
    return {
        "minimum": float(np.min(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q05": float(np.quantile(flat, 0.05)),
        "median": float(np.median(flat)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "maximum": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }


def run() -> None:
    if sha256(PREDECLARE) != FROZEN_PREDECLARE_SHA256:
        raise RuntimeError("Phase90 predeclaration hash mismatch")
    predeclare = json.loads(PREDECLARE.read_text(encoding="utf-8"))
    formula = predeclare["candidate_formula"]
    if float(formula["low_tail_quantile"]) != LOW_TAIL_QUANTILE:
        raise RuntimeError("quantile differs from Phase90 predeclaration")
    if float(predeclare["reference"]["symmetric_power_norm_boundary"]) != SYMMETRIC_BOUNDARY:
        raise RuntimeError("boundary differs from Phase90 predeclaration")

    folds, official_counts, actual_counts = p85.reconstruct_phase9_statistics()
    fold_rows: list[dict[str, Any]] = []
    reconstruction_errors = []
    for fold, stats in enumerate(folds):
        identity = np.ones_like(stats["A"], dtype=np.float64)
        baseline = p85.score_statistics(stats, identity)
        expected = p82.expected_phase9_fold(fold)
        reconstruction_errors.append({
            key: float(baseline[key] - expected[key])
            for key in ("c1_pas", "c2_pdp", "c3_nmse", "score")
        })
        scales, diagnostics = prediction_only_scales(stats)
        candidate = p85.score_statistics(stats, scales)
        inactive = scales == 1.0
        active = scales > 1.0
        weights = np.repeat(stats["weights"][:, None], 4, axis=1)
        weighted_active = float(
            np.sum(weights * active, dtype=np.float64) / np.sum(weights, dtype=np.float64)
        )
        # A future builder copies the P9 array first and assigns multiplication only
        # to active branches. Therefore every scale-one branch is bitwise unchanged.
        identity_verification = {
            "inactive_branches": int(np.sum(inactive)),
            "active_branches": int(np.sum(active)),
            "all_inactive_scales_exact_float64_one": bool(
                np.all(scales[inactive].view(np.uint64) == np.float64(1.0).view(np.uint64))
            ),
            "conditional_copy_policy": "copy P9; multiply only branches with scale>1",
            "bitwise_identity_guaranteed_for_scale_one_branches": True,
        }
        row = {
            "fold": fold,
            "scale_quantiles": quantiles(scales),
            "q_pas_quantiles": quantiles(diagnostics["q_pas"]),
            "q_pdp_quantiles": quantiles(diagnostics["q_pdp"]),
            "q_min_quantiles": quantiles(diagnostics["q"]),
            "active_ue_fraction": float(np.mean(active)),
            "weighted_active_ue_fraction": weighted_active,
            "identity_verification": identity_verification,
            "clamp_before": {
                "pas": clamp_report(stats, identity, "pas"),
                "pdp": clamp_report(stats, identity, "pdp"),
            },
            "clamp_after": {
                "pas": clamp_report(stats, scales, "pas"),
                "pdp": clamp_report(stats, scales, "pdp"),
            },
            "reference_phase9": baseline,
            "candidate": candidate,
            "score_delta": float(candidate["score"] - baseline["score"]),
            "component_deltas": {
                key: float(candidate[key] - baseline[key])
                for key in ("c1_pas", "c2_pdp", "c3_nmse")
            },
        }
        fold_rows.append(row)
        print(json.dumps({
            "stage": "fold", "fold": fold,
            "active_ue_fraction": row["active_ue_fraction"],
            "weighted_active_ue_fraction": weighted_active,
            "scale_quantiles": row["scale_quantiles"],
            "score_delta": row["score_delta"],
        }), flush=True)

    max_reconstruction_error = max(
        abs(value) for row in reconstruction_errors for value in row.values()
    )
    deltas = np.asarray([row["score_delta"] for row in fold_rows], dtype=np.float64)
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=0))
    lcb = mean - 0.75 * std
    gates = {
        "mean_at_least_0.00075": mean >= 0.00075,
        "at_least_three_positive": int(np.sum(deltas > 0.0)) >= 3,
        "at_least_four_nonnegative": int(np.sum(deltas >= 0.0)) >= 4,
        "minimum_not_below_minus_0.0001": float(np.min(deltas)) >= -0.0001,
        "lcb_mean_minus_0.75_std_strictly_positive": lcb > 0.0,
        "all_values_finite": bool(np.all(np.isfinite(deltas))),
        "phase9_reconstruction_within_0.000001": max_reconstruction_error <= 1e-6,
    }
    payload = {
        "predeclare": PREDECLARE.name,
        "predeclare_sha256": sha256(PREDECLARE),
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "configuration_modified_after_predeclaration": False,
        "source_or_heldout_truth_used_in_scale_rule": False,
        "test_channel_generated": False,
        "test_artifact_built": False,
        "uploaded": False,
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "zero_channel_outliers_excluded": int(np.sum(np.load(ROOT / "train_energy.npy") <= 0)),
        "phase9_reconstruction_errors": reconstruction_errors,
        "phase9_reconstruction_max_abs_error": float(max_reconstruction_error),
        "folds": fold_rows,
        "summary": {
            "score_deltas": deltas.tolist(),
            "mean_score_delta": mean,
            "population_std_score_delta": std,
            "lcb_mean_minus_0.75_population_std": lcb,
            "minimum_score_delta": float(np.min(deltas)),
            "positive_folds": int(np.sum(deltas > 0.0)),
            "nonnegative_folds": int(np.sum(deltas >= 0.0)),
            "gates": gates,
            "passed": all(gates.values()),
            "next_step": "report_only_no_test_build_or_upload",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "complete", **payload["summary"]}), flush=True)


if __name__ == "__main__":
    run()
