from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp
from phase8_anchor_retained_pas_resolution_validation import project


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BASE_PATH = ROOT / "Round2_Test_Channel_matched_phase6_delta2053.npy"
BASE_PAS_PATH = ROOT / "matched_phase6_milestone_physics_pas_band24_test.npy"
TARGET_PATH = ROOT / os.environ.get(
    "R2_PHASE10_TARGET", "phase10_robust125_primary_anchor_pas_band24_test.npy"
)
VALIDATION_PATH = ROOT / os.environ.get(
    "R2_PHASE10_VALIDATION", "phase10_robust125_primary_anchor_joint_validation.json"
)
TARGET_MANIFEST_PATH = ROOT / os.environ.get(
    "R2_PHASE10_TARGET_MANIFEST",
    "phase10_robust125_primary_anchor_test_pas_manifest.json",
)
OUTPUT_PATH = ROOT / os.environ.get(
    "R2_PHASE10_OUTPUT", "Round2_Test_Channel_phase10_robust125_anchor_pas.npy"
)
MANIFEST_PATH = ROOT / os.environ.get(
    "R2_PHASE10_MANIFEST", "phase10_robust125_anchor_pas_submission_manifest.json"
)
BANDS = 24
ITERATIONS = 4
BATCH_SIZE = 4
SAMPLE_INDICES = (0, 1, 17, 249, 499)


def log(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def validate_inputs(
    base: np.ndarray, base_pas: np.ndarray, target: np.ndarray
) -> None:
    expected_channel = (500, 256, 4, 192)
    expected_descriptor = (500, 256, 4, 24)
    if base.shape != expected_channel or base.dtype != np.complex64:
        raise RuntimeError(f"unexpected base: {base.shape} {base.dtype}")
    for name, value in (("base_pas", base_pas), ("target", target)):
        if value.shape != expected_descriptor or value.dtype != np.float32:
            raise RuntimeError(f"unexpected {name}: {value.shape} {value.dtype}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values")
    if np.any(target < 0):
        raise RuntimeError("PAS target contains negative values")


@torch.no_grad()
def generate() -> dict[int, str]:
    base = np.load(BASE_PATH, mmap_mode="r")
    base_pas = np.load(BASE_PAS_PATH, mmap_mode="r")
    target = np.load(TARGET_PATH, mmap_mode="r")
    validate_inputs(base, base_pas, target)
    output = np.lib.format.open_memmap(
        OUTPUT_PATH, mode="w+", dtype=np.complex64, shape=base.shape
    )
    sample_hashes: dict[int, str] = {}
    for start in range(0, len(base), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(base))
        p = torch.as_tensor(np.asarray(base[start:stop]).copy(), device=DEVICE)
        base_band = torch.as_tensor(
            np.asarray(base_pas[start:stop]).copy(), device=DEVICE
        )
        desired_band = torch.as_tensor(
            np.asarray(target[start:stop]).copy(), device=DEVICE
        )
        value = project(p, base_band, desired_band, BANDS, ITERATIONS)
        block = value.cpu().numpy().astype(np.complex64)
        output[start:stop] = block
        for index in SAMPLE_INDICES:
            if start <= index < stop:
                sample_hashes[index] = array_sha256(block[index - start])
        if stop % 50 == 0 or stop == len(base):
            output.flush()
            log("generate", done=stop)
    output.flush()
    del output
    if set(sample_hashes) != set(SAMPLE_INDICES):
        raise RuntimeError(f"missing generated sample hashes: {sample_hashes}")
    return sample_hashes


@torch.no_grad()
def independent_reload_audit(
    generated_sample_hashes: dict[int, str],
) -> dict[str, object]:
    base = np.load(BASE_PATH, mmap_mode="r")
    target = np.load(TARGET_PATH, mmap_mode="r")
    output = np.load(OUTPUT_PATH, mmap_mode="r")
    if output.shape != (500, 256, 4, 192) or output.dtype != np.complex64:
        raise RuntimeError(f"unexpected output layout: {output.shape} {output.dtype}")

    reload_hashes = {
        index: array_sha256(np.asarray(output[index])) for index in SAMPLE_INDICES
    }
    reload_match = all(
        reload_hashes[index] == generated_sample_hashes[index]
        for index in SAMPLE_INDICES
    )
    finite = True
    nonzero = True
    row_energy_min = float("inf")
    row_energy_max = 0.0
    pas_cosine_sum = 0.0
    pas_cosine_count = 0
    pdp_cosine_sum = 0.0
    pdp_cosine_count = 0
    channel_error_sum = 0.0
    channel_base_energy_sum = 0.0

    for start in range(0, len(output), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(output))
        base_block = np.asarray(base[start:stop]).copy()
        output_block = np.asarray(output[start:stop]).copy()
        finite &= bool(np.isfinite(output_block).all())
        energy = np.sum(np.abs(output_block) ** 2, axis=(1, 2, 3))
        nonzero &= bool(np.all(energy > 0))
        row_energy_min = min(row_energy_min, float(energy.min()))
        row_energy_max = max(row_energy_max, float(energy.max()))
        channel_error_sum += float(np.sum(np.abs(output_block - base_block) ** 2))
        channel_base_energy_sum += float(np.sum(np.abs(base_block) ** 2))

        x = torch.as_tensor(output_block, device=DEVICE)
        p = torch.as_tensor(base_block, device=DEVICE)
        pas = torch.abs(rp.bs_fft_torch(x)) ** 2
        pas /= torch.linalg.vector_norm(pas, dim=1, keepdim=True).clamp_min(1e-30)
        pas = pas.reshape(len(x), 256, 4, BANDS, 8).mean(-1)
        pas /= torch.linalg.vector_norm(pas, dim=1, keepdim=True).clamp_min(1e-30)
        desired = torch.as_tensor(
            np.asarray(target[start:stop]).copy(), device=DEVICE
        )
        pas_cosine = (pas * desired).sum(1) / (
            torch.linalg.vector_norm(pas, dim=1)
            * torch.linalg.vector_norm(desired, dim=1)
        ).clamp_min(1e-30)
        pas_cosine_sum += float(pas_cosine.sum(dtype=torch.float64))
        pas_cosine_count += pas_cosine.numel()

        base_pdp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
        output_pdp = torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2
        pdp_cosine = (base_pdp * output_pdp).sum(-1) / (
            torch.linalg.vector_norm(base_pdp, dim=-1)
            * torch.linalg.vector_norm(output_pdp, dim=-1)
        ).clamp_min(1e-30)
        pdp_cosine_sum += float(pdp_cosine.sum(dtype=torch.float64))
        pdp_cosine_count += pdp_cosine.numel()

    audit = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "bytes": OUTPUT_PATH.stat().st_size,
        "finite": finite,
        "all_test_rows_nonzero": nonzero,
        "row_energy_min": row_energy_min,
        "row_energy_max": row_energy_max,
        "pas_target_mean_cosine": pas_cosine_sum / pas_cosine_count,
        "phase6_pdp_mean_cosine": pdp_cosine_sum / pdp_cosine_count,
        "nmse_vs_phase6_channel": channel_error_sum / channel_base_energy_sum,
        "generated_sample_hashes": {
            str(index): generated_sample_hashes[index] for index in SAMPLE_INDICES
        },
        "reloaded_sample_hashes": {
            str(index): reload_hashes[index] for index in SAMPLE_INDICES
        },
        "independent_reload_sample_hashes_match": reload_match,
    }
    if not finite or not nonzero or not reload_match:
        raise RuntimeError(f"output audit failed: {audit}")
    return audit


def write_manifest(audit: dict[str, object]) -> dict[str, object]:
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    target_manifest = json.loads(TARGET_MANIFEST_PATH.read_text(encoding="utf-8"))
    output_hash = sha256(OUTPUT_PATH)
    base_manifest = json.loads(
        (ROOT / "matched_phase6_manifest.json").read_text(encoding="utf-8")
    )
    summary = validation["summary"]
    complement_variant = "joint_anchor_fold_deltas_vs_phase6" in summary
    if complement_variant:
        fold_deltas = summary["joint_anchor_fold_deltas_vs_phase6"]
        geometry_delta = summary["joint_anchor_geometry_weighted_delta"]
        locked_delta = summary["locked_edge_joint_anchor_delta"]
        lcb_delta = float(
            np.mean(fold_deltas) - 0.75 * np.std(fold_deltas)
        )
        variant_validation = {
            "frozen_before_evaluation": validation["frozen_before_evaluation"],
            "selection_or_tuning_performed": validation[
                "selection_or_tuning_performed"
            ],
            "joint_fold_deltas_vs_phase6": fold_deltas,
            "joint_geometry_weighted_delta": geometry_delta,
            "joint_min_fold_delta": float(min(fold_deltas)),
            "joint_lcb_delta": lcb_delta,
            "locked_edge_joint_delta": locked_delta,
            "fold_increments_vs_core_anchor": summary[
                "fold_increments_vs_core_anchor"
            ],
            "geometry_weighted_increment_vs_core_anchor": summary[
                "geometry_weighted_increment_vs_core_anchor"
            ],
            "locked_edge_increment_vs_core_anchor": summary[
                "locked_edge_increment_vs_core_anchor"
            ],
            "rotation_increments_vs_core_anchor": summary[
                "rotation_increments_vs_core_anchor"
            ],
            "rotation_positive_count": summary["rotation_positive_count"],
            "all_folds_positive": bool(np.all(np.asarray(fold_deltas) > 0)),
            "all_fold_non_decreasing_vs_core_anchor": summary[
                "all_fold_non_decreasing"
            ],
            "fixed_confirmation_passed": summary["fixed_confirmation_passed"],
        }
        target_record = target_manifest["final_joint_pas_target"]
    else:
        geometry_delta = summary["joint_geometry_weighted_delta"]
        locked_delta = summary["locked_edge_joint_delta"]
        variant_validation = {
            "frozen_before_evaluation": validation["frozen_before_evaluation"],
            "joint_fold_deltas_vs_phase6": summary[
                "joint_fold_deltas_vs_phase6"
            ],
            "joint_geometry_weighted_delta": geometry_delta,
            "joint_min_fold_delta": summary["joint_min_fold_delta"],
            "joint_lcb_delta": summary["joint_lcb_delta"],
            "locked_edge_joint_delta": locked_delta,
            "locked_edge_anchor_increment": summary[
                "locked_edge_anchor_increment"
            ],
            "inner_anchor_increments": summary["inner_anchor_increments"],
            "inner_anchor_positive_count": summary[
                "inner_anchor_positive_count"
            ],
            "all_folds_positive": summary["all_folds_positive"],
        }
        target_record = target_manifest["joint_pas_target"]
    manifest = {
        "output": OUTPUT_PATH.name,
        **audit,
        "sha256": output_hash,
        "base": {
            "path": BASE_PATH.name,
            "sha256": base_manifest["sha256"],
            "cumulative_delta_vs_matched_v3": base_manifest[
                "cumulative_delta_vs_matched_v3"
            ],
        },
        "pas_target": target_record,
        "validation": variant_validation,
        "local_cumulative_delta_estimates_vs_matched_v3": {
            "geometry_weighted": base_manifest["cumulative_delta_vs_matched_v3"]
            + geometry_delta,
            "locked_edge": base_manifest["cumulative_delta_vs_matched_v3"]
            + locked_delta,
        },
        "model": {
            "channel_base": "Phase6 official-best base",
            "pas_target": TARGET_PATH.name,
            "pas_bands": BANDS,
            "alternating_projection_iterations": ITERATIONS,
            "pas_ratio_clip": [0.25, 4.0],
            "pdp_constraint": "the Phase6 base PDP is the sole PDP constraint each iteration; no new PDP target",
            "pdp_correction_power": 1.5,
            "final_projection_step": "PAS",
            "batch_size": BATCH_SIZE,
        },
        "training_rows_used": 3738,
        "zero_channel_outliers_removed": 262,
        "upload_status": "not uploaded",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("manifest_saved", path=MANIFEST_PATH.name, sha256=output_hash)
    return manifest


def run() -> None:
    generated_hashes = generate()
    audit = independent_reload_audit(generated_hashes)
    manifest = write_manifest(audit)
    log(
        "complete",
        output=manifest["output"],
        sha256=manifest["sha256"],
        finite=manifest["finite"],
        all_test_rows_nonzero=manifest["all_test_rows_nonzero"],
        reload_match=manifest["independent_reload_sample_hashes_match"],
    )


if __name__ == "__main__":
    run()
