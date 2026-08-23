from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r2_pipeline as rp
import phase18_orthogonal_p9_reverse_secant_validation as phase18
import phase85_metric_epsilon_perue as phase85
import phase89_g56_full192_then_antip10_pas050_validation as phase89
import phase90_symmetric_clampfloor_perue as phase90
from phase8_anchor_retained_pas_resolution_validation import anchor_prediction, project
from phase8_anchor_retained_pas_screen import horizontal_shifts, mapped_anchors
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_anchor_joint_channel_validation import project_joint
from phase10_calibrated_pas_residual_joint_validation import update_weighted_scores
from phase8_anchor_local_gate_channel_validation import components


ROOT = Path(__file__).resolve().parent
PREDECLARE = ROOT / "phase93_g56_antip10_plus_symmetric_clamp_predeclared.json"
OUTPUT = ROOT / "phase93_g56_antip10_plus_symmetric_clamp_validation.json"
FROZEN_PREDECLARE_SHA256 = "bfc9c5dd88ce1a4d2a767515a906657cabd252d7cadf959d5fdfee4cf60b6734"
DEVICE = torch.device("cuda")
ACTIVE_GROUPS = (5, 6)
DOSE = 0.20
BATCH = 2
NAMES = ("p9", "phase40", "phase89", "phase93")
SOURCE_HASHES = {
    "phase89_script": (
        "phase89_g56_full192_then_antip10_pas050_validation.py",
        "c171767b00d60de8bedde5af276de6d10fb4a97e13810e2db7c483a081d3b796",
    ),
    "phase89_result": (
        "phase89_g56_full192_then_antip10_pas050_validation.json",
        "ce3c3f7574daa23c2d6a9f70f3335117ee8074d2440fa2842c4684ff909a49c9",
    ),
    "phase90_predeclare": (
        "phase90_symmetric_clampfloor_perue_predeclared.json",
        "b19d1da428860a20d54cce6f72f43e827219875796255eca22177fa23a68314c",
    ),
    "phase90_script": (
        "phase90_symmetric_clampfloor_perue.py",
        "43e89b99bd41473f754561ac95da9a3f1f3bb26f2969d8d0f0c5cbd2d4ce2835",
    ),
    "phase90_result": (
        "phase90_symmetric_clampfloor_perue.json",
        "7134414fc12ef5c2162f7d24ec009492d79719c86b3d20efc4039895bd408c52",
    ),
    "phase85_statistics": (
        "phase85_metric_epsilon_perue.py",
        "3e0664cfd8e52b1a048481bfc0746a66511203cad94cdeb30417000a5377587e",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def audit_sources() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, (name, expected) in SOURCE_HASHES.items():
        actual = sha256(ROOT / name)
        output[key] = {
            "path": name,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        }
    if not all(row["match"] for row in output.values()):
        raise RuntimeError("Phase93 frozen source audit failed")
    return output


def component_error(actual: dict[str, float], expected: dict[str, float]) -> float:
    return max(
        abs(float(actual[key]) - float(expected[key]))
        for key in ("c1_pas", "c2_pdp", "c3_nmse", "score")
    )


def scale_quantiles(scales: np.ndarray) -> dict[str, float]:
    values = scales.reshape(-1)
    return {
        "minimum": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


@torch.no_grad()
def evaluate_fold(
    fold: int,
    row: dict[str, Any],
    channel: np.ndarray,
    anchor192: np.ndarray,
    phase89_reference: dict[str, Any],
    phase90_reference: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    initializer_file = np.load(
        ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r"
    )
    labels = row["labels"][row["query_local"]]
    accumulator = np.zeros((len(NAMES), 6), dtype=np.float64)
    group_accumulator = np.zeros((len(NAMES), 11, 6), dtype=np.float64)
    statistics_batches: list[dict[str, np.ndarray]] = []
    scale_rows = []
    active_scale_branches = 0
    inactive_scale_branches = 0
    scale_one_identity = True
    finite_nonzero = True

    for start in range(0, len(row["query"]), BATCH):
        stop = min(start + BATCH, len(row["query"]))
        local_query = row["query_local"][start:stop]
        batch_labels = labels[start:stop]
        initializer = torch.as_tensor(
            np.asarray(initializer_file[local_query]).copy(), device=DEVICE
        )
        truth = torch.as_tensor(
            np.asarray(channel[row["query"][start:stop]]).copy(), device=DEVICE
        )
        truth_pas = torch.abs(rp.bs_fft_torch(truth)) ** 2
        truth_pdp = torch.abs(torch.fft.fft(truth, dim=-1, norm="ortho")) ** 2
        base_pas = torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE)

        p9 = project_joint(
            initializer,
            base_pas,
            torch.as_tensor(row["phase9_pas"][start:stop].copy(), device=DEVICE),
            torch.as_tensor(row["phase9_pdp"][start:stop].copy(), device=DEVICE),
            12,
        )
        p10 = project_joint(
            initializer,
            base_pas,
            torch.as_tensor(row["phase10_pas"][start:stop].copy(), device=DEVICE),
            torch.as_tensor(row["base_pdp"][start:stop].copy(), device=DEVICE),
            4,
        )

        g56 = p9.clone()
        active_group_np = np.isin(batch_labels, ACTIVE_GROUPS)
        if np.any(active_group_np):
            active_group = torch.as_tensor(active_group_np, device=DEVICE)
            base192 = phase89.full_pas(p9[active_group])
            retained192 = torch.as_tensor(
                anchor192[start:stop][active_group_np].copy(), device=DEVICE
            )
            desired192 = phase89.normalize_dim1(
                (1.0 - DOSE) * base192 + DOSE * retained192
            )
            g56[active_group] = project(
                p9[active_group], base192, desired192, 192, 12
            )

        direction = phase89.phase40_direction(p9, p10)
        phase40_value = phase89.apply_frozen_phase40(p9, direction)
        phase89_value = phase89.apply_frozen_phase40(g56, direction)

        stats = phase85.batch_statistics(p9, truth)
        statistics_batches.append(stats)
        scales, _ = phase90.prediction_only_scales(
            {
                "pas_pnorm": stats["pas_pnorm"],
                "pdp_pnorm": stats["pdp_pnorm"],
            }
        )
        scale_rows.append(scales)
        phase93_value = phase89_value.clone()
        active_scale = scales > 1.0
        inactive_scale = ~active_scale
        active_scale_branches += int(np.sum(active_scale))
        inactive_scale_branches += int(np.sum(inactive_scale))
        for local_row, ue in np.argwhere(active_scale):
            phase93_value[int(local_row), :, int(ue), :] = (
                phase89_value[int(local_row), :, int(ue), :] * float(scales[local_row, ue])
            )
        for local_row, ue in np.argwhere(inactive_scale):
            scale_one_identity &= bool(
                torch.equal(
                    phase93_value[int(local_row), :, int(ue), :],
                    phase89_value[int(local_row), :, int(ue), :],
                )
            )

        weights = torch.as_tensor(
            row["weights"][start:stop].astype(np.float32), device=DEVICE
        )
        for index, value in enumerate((p9, phase40_value, phase89_value, phase93_value)):
            block = value.detach().cpu().numpy()
            finite_nonzero &= bool(
                np.isfinite(block).all()
                and np.all(np.sum(np.abs(block) ** 2, axis=(1, 2, 3)) > 0)
            )
            update_weighted_scores(
                accumulator,
                group_accumulator,
                index,
                value,
                truth,
                truth_pas,
                truth_pdp,
                weights,
                batch_labels,
            )

    standard = {
        name: components(accumulator[index]) for index, name in enumerate(NAMES)
    }
    stats = phase85.concatenate_batches(statistics_batches)
    stats["weights"] = row["weights"].astype(np.float32).astype(np.float64)
    all_scales, diagnostics = phase90.prediction_only_scales(
        {
            "pas_pnorm": stats["pas_pnorm"],
            "pdp_pnorm": stats["pdp_pnorm"],
        }
    )
    batched_scales = np.concatenate(scale_rows)
    scale_reproduction_error = float(np.max(np.abs(all_scales - batched_scales)))
    analytic_p9 = phase85.score_statistics(stats, np.ones_like(all_scales))
    analytic_phase90 = phase85.score_statistics(stats, all_scales)

    expected_standard = phase89_reference["standard_official_rectangle_weighting"]
    reproduction_errors = {
        "p9": component_error(standard["p9"], expected_standard["p9"]),
        "phase40": component_error(standard["phase40"], expected_standard["phase40"]),
        "phase89": component_error(standard["phase89"], expected_standard["combined"]),
        "phase90_reference_p9": component_error(
            analytic_p9, phase90_reference["reference_phase9"]
        ),
        "phase90_candidate": component_error(
            analytic_phase90, phase90_reference["candidate"]
        ),
        "phase90_batched_scale": scale_reproduction_error,
    }
    reproduction_checks = {
        "p9_lte_1e-10": reproduction_errors["p9"] <= 1e-10,
        "phase40_lte_1e-10": reproduction_errors["phase40"] <= 1e-10,
        "phase89_lte_1e-10": reproduction_errors["phase89"] <= 1e-10,
        "phase90_lte_1e-8": max(
            reproduction_errors["phase90_reference_p9"],
            reproduction_errors["phase90_candidate"],
            reproduction_errors["phase90_batched_scale"],
        ) <= 1e-8,
    }
    delta = standard["phase93"]["score"] - standard["phase40"]["score"]
    weights2d = np.repeat(row["weights"][:, None], 4, axis=1)
    weighted_active_scale_fraction = float(
        np.sum(weights2d * (all_scales > 1.0), dtype=np.float64)
        / np.sum(weights2d, dtype=np.float64)
    )
    result = {
        "fold": fold,
        "query_rows": len(row["query"]),
        "standard_official_rectangle_weighting": {
            name: {
                **value,
                **({"delta_vs_phase40": delta} if name == "phase93" else {}),
            }
            for name, value in standard.items()
        },
        "phase90_scales_from_original_p9": {
            "quantiles": scale_quantiles(all_scales),
            "q_pas_quantiles": scale_quantiles(diagnostics["q_pas"]),
            "q_pdp_quantiles": scale_quantiles(diagnostics["q_pdp"]),
            "active_branches": active_scale_branches,
            "inactive_branches": inactive_scale_branches,
            "weighted_active_fraction": weighted_active_scale_fraction,
            "all_scale_one_branches_bitwise_equal_phase89": scale_one_identity,
        },
        "phase90_analytic_reproduction": {
            "p9": analytic_p9,
            "candidate": analytic_phase90,
        },
        "reproduction_errors": reproduction_errors,
        "reproduction_checks": reproduction_checks,
        "finite_nonzero": finite_nonzero,
    }
    print(
        json.dumps(
            {
                "stage": "fold",
                "fold": fold,
                "delta_vs_phase40": delta,
                "active_scale_branches": active_scale_branches,
                "reproductions": reproduction_checks,
            }
        ),
        flush=True,
    )
    return result, group_accumulator


def main() -> None:
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError("Phase93 output exists; refusing to overwrite")
    if sha256(PREDECLARE) != FROZEN_PREDECLARE_SHA256:
        raise RuntimeError("Phase93 predeclare hash mismatch")
    predeclare = json.loads(PREDECLARE.read_text(encoding="utf-8"))
    if (
        predeclare["phase"] != 93
        or not predeclare["frozen_before_any_phase92_or_phase93_score_was_read"]
        or predeclare["selection_disclosure"]["phase92_result_known_at_freeze"]
        or predeclare["selection_disclosure"]["dose_or_parameter_search"]
        or float(predeclare["fixed_composition"]["phase90_scale_rule"]["power_norm_low_tail_quantile"]) != phase90.LOW_TAIL_QUANTILE
        or float(predeclare["fixed_composition"]["phase90_scale_rule"]["symmetric_boundary"]) != phase90.SYMMETRIC_BOUNDARY
        or tuple(predeclare["fixed_composition"]["phase90_scale_rule"]["clip"]) != phase90.SCALE_BOUNDS
    ):
        raise RuntimeError("Phase93 predeclare invariant failed")
    source_audit = audit_sources()
    phase89_data = json.loads(
        (ROOT / SOURCE_HASHES["phase89_result"][0]).read_text(encoding="utf-8")
    )
    phase90_data = json.loads(
        (ROOT / SOURCE_HASHES["phase90_result"][0]).read_text(encoding="utf-8")
    )

    folds, channel, energy, official_counts, actual_counts = phase18.prepare_all_targets()
    pos, _, reloaded_energy = rp.load_data()
    if not np.array_equal(energy, reloaded_energy):
        raise RuntimeError("Energy reload mismatch")
    if int(np.sum(energy <= 0)) != 262:
        raise RuntimeError("Zero-outlier invariant failed")
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)
    shifts = horizontal_shifts(pos)

    anchors192 = []
    for fold, row in enumerate(folds):
        retained = mapped_anchors(
            pos, row["val"], row["labels"], actual_fraction, official_counts
        )
        truth192 = np.load(
            ROOT / f"phase8_anchor_retained_fold{fold}_truth_pas_band192.npy",
            mmap_mode="r",
        )
        anchors192.append(
            anchor_prediction(
                pos[row["val"], :2],
                row["labels"],
                shifts[row["val"]],
                retained,
                row["query_local"],
                np.asarray(truth192),
                "horizontal",
            )
        )
        print(json.dumps({"stage": "anchor192", "fold": fold}), flush=True)

    fold_results = []
    group_accumulators = []
    for fold, row in enumerate(folds):
        result, group_accumulator = evaluate_fold(
            fold,
            row,
            channel,
            anchors192[fold],
            phase89_data["folds"][fold],
            phase90_data["folds"][fold],
        )
        fold_results.append(result)
        group_accumulators.append(group_accumulator)

    deltas = np.asarray(
        [
            row["standard_official_rectangle_weighting"]["phase93"]["delta_vs_phase40"]
            for row in fold_results
        ],
        dtype=np.float64,
    )
    mean = float(np.mean(deltas))
    std = float(np.std(deltas))
    lcb = mean - 0.75 * std
    positive = int(np.sum(deltas > 0))
    minimum = float(np.min(deltas))
    all_reproductions = all(
        all(row["reproduction_checks"].values()) for row in fold_results
    )
    identity = all(
        row["phase90_scales_from_original_p9"][
            "all_scale_one_branches_bitwise_equal_phase89"
        ]
        for row in fold_results
    )
    finite_nonzero = all(row["finite_nonzero"] for row in fold_results)
    checks = {
        "mean_at_least_0.00075": mean >= 0.00075,
        "at_least_4_of_5_positive": positive >= 4,
        "minimum_at_least_minus_0.0002": minimum >= -0.0002,
        "mean_minus_0.75std_strictly_positive": lcb > 0,
        "all_reproductions_pass": all_reproductions,
        "scale_one_branches_bitwise_identity": identity,
        "all_finite_nonzero": finite_nonzero,
    }
    passed = all(checks.values())

    group_deltas: dict[str, list[float]] = {}
    for group in range(11):
        per_fold = []
        for accumulator in group_accumulators:
            phase93_value = components(accumulator[NAMES.index("phase93"), group])
            phase40_value = components(accumulator[NAMES.index("phase40"), group])
            per_fold.append(phase93_value["score"] - phase40_value["score"])
        group_deltas[str(group)] = per_fold

    payload = {
        "phase": 93,
        "predeclare": PREDECLARE.name,
        "predeclare_sha256": sha256(PREDECLARE),
        "validation_script_sha256_before_output": sha256(Path(__file__).resolve()),
        "phase92_result_used": False,
        "source_audit": source_audit,
        "zero_outliers_removed": int(np.sum(energy <= 0)),
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "folds": fold_results,
        "group_deltas_vs_phase40": group_deltas,
        "summary": {
            "fold_deltas_vs_phase40": deltas.tolist(),
            "mean": mean,
            "population_std": std,
            "lcb_mean_minus_0.75std": lcb,
            "positive_folds": positive,
            "minimum": minimum,
            "checks": checks,
            "pass": passed,
        },
        "separate_test_build_plan": {
            "permitted": passed,
            "actual_build_performed": False,
            "step1": "Construct the exact Phase89 500-row channel from existing P9/P10 and retained anchors.",
            "step2": "Compute Phase90 scales solely from each original P9 row/UE using the frozen 1% low-tail norm rule.",
            "step3": "Copy Phase89, multiplying only UE branches with scale>1; preserve every scale-one branch bitwise.",
            "test_truth_required": False,
        },
        "decision": "permit_separate_test_build" if passed else "retire_no_test_no_adjustment",
        "test_prediction_generated": False,
        "uploaded": False,
        "elapsed_seconds": time.time() - started,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": "complete",
                "decision": payload["decision"],
                "summary": payload["summary"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
