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
from phase8_anchor_retained_pas_resolution_validation import (
    anchor_prediction,
    project,
)
from phase8_anchor_retained_pas_screen import horizontal_shifts, mapped_anchors
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_anchor_joint_channel_validation import project_joint
from phase10_calibrated_pas_residual_joint_validation import update_weighted_scores
from phase8_anchor_local_gate_channel_validation import components


ROOT = Path(__file__).resolve().parent
PREDECLARE = ROOT / "phase89_g56_full192_then_antip10_pas050_predeclared.json"
OUTPUT = ROOT / "phase89_g56_full192_then_antip10_pas050_validation.json"
DEVICE = torch.device("cuda")
ACTIVE_GROUPS = (5, 6)
FULL192_DOSE = 0.20
ANTI_ETA = 0.50
PAS_EPS = 1e-4 / 256
BATCH = 2
NAMES = ("p9", "phase79_g56", "phase40", "combined")
REPRODUCTION_TOLERANCE = 2e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def normalize_dim1(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(1e-30)


def full_pas(value: torch.Tensor) -> torch.Tensor:
    return normalize_dim1(torch.abs(rp.bs_fft_torch(value)) ** 2)


def band24(value: torch.Tensor) -> torch.Tensor:
    power = normalize_dim1(torch.abs(rp.bs_fft_torch(value)) ** 2)
    return normalize_dim1(power.reshape(len(value), 256, 4, 24, 8).mean(-1))


def phase40_direction(p9: torch.Tensor, p10: torch.Tensor) -> torch.Tensor:
    return torch.log((band24(p10) + PAS_EPS) / (band24(p9) + PAS_EPS)).clamp(-2, 2)


def apply_frozen_phase40(
    current: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    current_band = band24(current)
    desired = normalize_dim1(current_band * torch.exp(-ANTI_ETA * direction))
    spectrum = rp.bs_fft_torch(current)
    power = torch.abs(spectrum) ** 2
    target = power * ((desired + PAS_EPS) / (current_band + PAS_EPS)).repeat_interleave(8, dim=3)
    return rp.bs_ifft_torch(
        spectrum * torch.sqrt(target / power.clamp_min(1e-30))
    )


def source_hash_audit(predeclare: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, row in predeclare["source_audit"].items():
        path = ROOT / row["path"]
        actual = sha256(path)
        output[key] = {
            "path": row["path"],
            "expected_sha256": row["sha256"],
            "actual_sha256": actual,
            "match": actual == row["sha256"],
        }
    if not all(row["match"] for row in output.values()):
        raise RuntimeError("Frozen source hash audit failed")
    return output


def maximum_component_error(actual: dict[str, float], expected: dict[str, float]) -> float:
    return max(
        abs(float(actual[key]) - float(expected[key]))
        for key in ("c1_pas", "c2_pdp", "c3_nmse", "score")
    )


@torch.no_grad()
def evaluate_fold(
    fold: int,
    row: dict[str, Any],
    channel: np.ndarray,
    anchor192: np.ndarray,
    phase79_reference: dict[str, Any],
    phase40_reference: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
    labels = row["labels"][row["query_local"]]
    standard_accumulator = np.zeros((len(NAMES), 6), dtype=np.float64)
    standard_group = np.zeros((len(NAMES), 11, 6), dtype=np.float64)
    calibrated_accumulator = np.zeros((len(NAMES), 6), dtype=np.float64)
    calibrated_group = np.zeros((len(NAMES), 11, 6), dtype=np.float64)
    inactive_identity_error = 0.0
    all_finite_nonzero = True
    active_rows = 0

    for start in range(0, len(row["query"]), BATCH):
        stop = min(start + BATCH, len(row["query"]))
        local_query = row["query_local"][start:stop]
        batch_labels = labels[start:stop]
        initializer = torch.as_tensor(
            np.asarray(prediction[local_query]).copy(), device=DEVICE
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
        active_np = np.isin(batch_labels, ACTIVE_GROUPS)
        active_rows += int(np.sum(active_np))
        if np.any(active_np):
            active = torch.as_tensor(active_np, device=DEVICE)
            p9_full = full_pas(p9[active])
            retained = torch.as_tensor(
                anchor192[start:stop][active_np].copy(), device=DEVICE
            )
            desired = normalize_dim1(
                (1.0 - FULL192_DOSE) * p9_full + FULL192_DOSE * retained
            )
            g56[active] = project(p9[active], p9_full, desired, 192, 12)

        direction = phase40_direction(p9, p10)
        phase40 = apply_frozen_phase40(p9, direction)
        combined = apply_frozen_phase40(g56, direction)
        inactive_np = ~active_np
        if np.any(inactive_np):
            inactive = torch.as_tensor(inactive_np, device=DEVICE)
            inactive_identity_error = max(
                inactive_identity_error,
                float(torch.max(torch.abs(combined[inactive] - phase40[inactive])).cpu()),
            )

        values = (p9, g56, phase40, combined)
        standard_weights = torch.as_tensor(
            row["weights"][start:stop].astype(np.float32), device=DEVICE
        )
        calibrated_weights = torch.as_tensor(
            row["calibrated_weights"][start:stop].astype(np.float32), device=DEVICE
        )
        for index, value in enumerate(values):
            block = value.detach().cpu().numpy()
            all_finite_nonzero &= bool(
                np.isfinite(block).all()
                and np.all(np.sum(np.abs(block) ** 2, axis=(1, 2, 3)) > 0)
            )
            update_weighted_scores(
                standard_accumulator,
                standard_group,
                index,
                value,
                truth,
                truth_pas,
                truth_pdp,
                standard_weights,
                batch_labels,
            )
            update_weighted_scores(
                calibrated_accumulator,
                calibrated_group,
                index,
                value,
                truth,
                truth_pas,
                truth_pdp,
                calibrated_weights,
                batch_labels,
            )

    standard = {
        name: components(standard_accumulator[index])
        for index, name in enumerate(NAMES)
    }
    calibrated = {
        name: components(calibrated_accumulator[index])
        for index, name in enumerate(NAMES)
    }
    expected_phase40 = phase40_reference["variants"]["anti_pas_direct_0.50"]
    reproductions = {
        "phase9_standard_max_component_error_vs_phase79": maximum_component_error(
            standard["p9"], phase79_reference["baseline_phase9"]
        ),
        "phase79_standard_max_component_error": maximum_component_error(
            standard["phase79_g56"], phase79_reference["candidate"]
        ),
        "phase9_calibrated_max_component_error_vs_phase39b": maximum_component_error(
            calibrated["p9"], phase40_reference["p9"]
        ),
        "phase40_calibrated_max_component_error_vs_phase39b": maximum_component_error(
            calibrated["phase40"], expected_phase40
        ),
    }
    reproductions["pass"] = bool(
        max(float(value) for key, value in reproductions.items() if key != "pass")
        <= REPRODUCTION_TOLERANCE
    )
    result = {
        "fold": fold,
        "query_rows": len(row["query"]),
        "active_rows": active_rows,
        "standard_official_rectangle_weighting": {
            name: {
                **value,
                **(
                    {"delta_vs_phase40": value["score"] - standard["phase40"]["score"]}
                    if name == "combined"
                    else {}
                ),
            }
            for name, value in standard.items()
        },
        "calibrated_audit_weighting": calibrated,
        "reproductions": reproductions,
        "inactive_combined_vs_phase40_max_abs_error": inactive_identity_error,
        "finite_nonzero": all_finite_nonzero,
    }
    print(
        json.dumps(
            {
                "stage": "fold",
                "fold": fold,
                "active_rows": active_rows,
                "delta_vs_phase40": result["standard_official_rectangle_weighting"]["combined"]["delta_vs_phase40"],
                "reproduction_pass": reproductions["pass"],
            }
        ),
        flush=True,
    )
    return result, standard_group


def main() -> None:
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError("Phase89 output already exists; refusing to overwrite first-run audit")
    predeclare = json.loads(PREDECLARE.read_text(encoding="utf-8"))
    source_audit = source_hash_audit(predeclare)
    phase79_data = json.loads(
        (ROOT / "phase79_g5g6_full192_on_p9.json").read_text(encoding="utf-8")
    )
    phase40_data = json.loads(
        (ROOT / "phase39b_actual_antip10_direct_pas_exact_validation.json").read_text(
            encoding="utf-8"
        )
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

    anchors_by_fold: list[np.ndarray] = []
    for fold, row in enumerate(folds):
        anchors = mapped_anchors(
            pos, row["val"], row["labels"], actual_fraction, official_counts
        )
        truth192 = np.load(
            ROOT / f"phase8_anchor_retained_fold{fold}_truth_pas_band192.npy",
            mmap_mode="r",
        )
        anchors_by_fold.append(
            anchor_prediction(
                pos[row["val"], :2],
                row["labels"],
                shifts[row["val"]],
                anchors,
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
            anchors_by_fold[fold],
            phase79_data["folds"][fold],
            phase40_data["folds"][fold],
        )
        fold_results.append(result)
        group_accumulators.append(group_accumulator)

    deltas = np.asarray(
        [
            row["standard_official_rectangle_weighting"]["combined"]["delta_vs_phase40"]
            for row in fold_results
        ],
        dtype=np.float64,
    )
    mean = float(np.mean(deltas))
    std = float(np.std(deltas))
    lcb = mean - 0.75 * std
    positive = int(np.sum(deltas > 0))
    minimum = float(np.min(deltas))
    reproduction_pass = all(row["reproductions"]["pass"] for row in fold_results)
    finite_nonzero = all(row["finite_nonzero"] for row in fold_results)
    inactive_identity_max = max(
        row["inactive_combined_vs_phase40_max_abs_error"] for row in fold_results
    )
    checks = {
        "mean_at_least_0.0004": mean >= 0.0004,
        "at_least_4_of_5_positive": positive >= 4,
        "mean_minus_0.75std_strictly_positive": lcb > 0,
        "minimum_at_least_minus_0.0002": minimum >= -0.0002,
        "all_reproductions_pass": reproduction_pass,
        "all_finite_nonzero": finite_nonzero,
        "inactive_rows_exactly_phase40": inactive_identity_max == 0.0,
    }
    passed = all(checks.values())

    group_deltas: dict[str, list[float]] = {}
    for group in ACTIVE_GROUPS:
        values = []
        for accumulator in group_accumulators:
            combined = components(accumulator[NAMES.index("combined"), group])
            phase40 = components(accumulator[NAMES.index("phase40"), group])
            values.append(combined["score"] - phase40["score"])
        group_deltas[str(group)] = values

    payload = {
        "phase": 89,
        "protocol": "fixed g5/g6 full192 dose0.20 first, then original-P9/P10 anti-P10 direct PAS eta0.50",
        "predeclare": PREDECLARE.name,
        "predeclare_sha256": sha256(PREDECLARE),
        "validation_script_sha256_before_output": sha256(Path(__file__).resolve()),
        "source_hash_audit": source_audit,
        "zero_outliers_removed": int(np.sum(energy <= 0)),
        "official_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "folds": fold_results,
        "active_group_unweighted_deltas_vs_phase40": group_deltas,
        "summary": {
            "fold_deltas_vs_phase40_official_rectangle_weighting": deltas.tolist(),
            "mean": mean,
            "population_std": std,
            "lcb_mean_minus_0.75std": lcb,
            "positive_folds": positive,
            "minimum": minimum,
            "inactive_identity_max_abs_error": inactive_identity_max,
            "checks": checks,
            "pass": passed,
        },
        "buildability": {
            "deterministic": True,
            "test_truth_required": False,
            "test_group_labels": "rp.official_island_labels(Round2_Test_Pos.npy)",
            "step1": "Build Phase79 retained-anchor horizontal full192 target for test groups 5/6 only and project P9 at fixed dose0.20.",
            "step2": "Compute the clipped log direction from the existing P9 and P10 test channels, then apply eta0.50 once to the step1 channel.",
            "nonactive_rows": "Exactly Phase40 by construction and verified exact in all validation batches.",
            "permitted_if_pass": passed,
        },
        "decision": "permit_separate_test_build" if passed else "terminate_no_test",
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
