from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r2_pipeline as rp
import phase89_g56_full192_then_antip10_pas050_validation as phase89
import phase90_symmetric_clampfloor_perue as phase90
from phase8_anchor_retained_pas_resolution_validation import anchor_prediction, project
from phase8_anchor_retained_pas_screen import horizontal_shifts


ROOT = Path(__file__).resolve().parent
TEMP = Path(os.environ["TEMP"])
P9 = ROOT / "Round2_Test_Channel_matched_phase9_buildable_anchor_joint.npy"
P10 = ROOT / "Round2_Test_Channel_phase10_core125_complement100_anchor_pas.npy"
PHASE40 = TEMP / "Round2_Test_Channel_phase40_p9_actual_antip10_pas050.npy"
OUTPUT = TEMP / "Round2_Test_Channel_phase93_g56_antip10_plus_symmetric_clamp.npy"
BUILDING = OUTPUT.with_name(OUTPUT.stem + ".building.npy")
MANIFEST = ROOT / "phase93_g56_antip10_plus_symmetric_clamp_submission_manifest.json"
DEVICE = torch.device("cuda")
BATCH = 2
ACTIVE_GROUPS = (5, 6)
DOSE = 0.20
EXPECTED_SHAPE = (500, 256, 4, 192)
EXPECTED_DTYPE = np.dtype(np.complex64)
EXPECTED_BYTES = 786_432_128
EXPECTED_HASHES = {
    "phase93_predeclare": (
        ROOT / "phase93_g56_antip10_plus_symmetric_clamp_predeclared.json",
        "bfc9c5dd88ce1a4d2a767515a906657cabd252d7cadf959d5fdfee4cf60b6734",
    ),
    "phase93_validation": (
        ROOT / "phase93_g56_antip10_plus_symmetric_clamp_validation.json",
        "5861ebe1ecb65c8783abdefac4cc05e135e973c977606a606fb4f8ebe30f9eee",
    ),
    "phase89_script": (
        ROOT / "phase89_g56_full192_then_antip10_pas050_validation.py",
        "c171767b00d60de8bedde5af276de6d10fb4a97e13810e2db7c483a081d3b796",
    ),
    "phase90_script": (
        ROOT / "phase90_symmetric_clampfloor_perue.py",
        "43e89b99bd41473f754561ac95da9a3f1f3bb26f2969d8d0f0c5cbd2d4ce2835",
    ),
    "p9": (
        P9,
        "b00817e8741a5cefc92ba376bde8875ecaa869868410886777609882ccf66a0c",
    ),
    "p10": (
        P10,
        "4ae82162723dbc61eed349ff0a8477b0432c6ce5c19e515a48815383757ac966",
    ),
    "phase40": (
        PHASE40,
        "4876ba6a39b0c9e9c1309d08615e2c3788f4149c4c2a594346393fab4e60a740",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    view = memoryview(np.ascontiguousarray(value)).cast("B")
    for start in range(0, len(view), 16 << 20):
        digest.update(view[start : start + (16 << 20)])
    return digest.hexdigest()


def audit_sources() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, (path, expected) in EXPECTED_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        output[key] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        }
        print(json.dumps({"stage": "source_hash", "source": key, "match": actual == expected}), flush=True)
    if not all(row["match"] for row in output.values()):
        raise RuntimeError("Frozen source hash mismatch")
    return output


def validate_source_array(path: Path) -> np.memmap:
    value = np.load(path, mmap_mode="r")
    if value.shape != EXPECTED_SHAPE or value.dtype != EXPECTED_DTYPE:
        raise RuntimeError(f"Unexpected source {path}: {value.shape} {value.dtype}")
    if path.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(f"Unexpected source byte size {path}: {path.stat().st_size}")
    return value


def official_train_labels(
    pos: np.ndarray,
    energy: np.ndarray,
    test_pos: np.ndarray,
    test_labels: np.ndarray,
) -> np.ndarray:
    groups = sorted(int(group) for group in np.unique(test_labels))
    boxes = {
        group: (
            test_pos[test_labels == group, :2].min(0),
            test_pos[test_labels == group, :2].max(0),
        )
        for group in groups
    }
    labels = np.full(len(pos), -1, dtype=np.int64)
    for index in np.flatnonzero(energy > 0):
        for group in groups:
            lo, hi = boxes[group]
            if np.all(pos[index, :2] >= lo) and np.all(pos[index, :2] <= hi):
                labels[index] = group
                break
    return labels


@torch.no_grad()
def full_pas_descriptors(channel: np.ndarray, rows: np.ndarray) -> np.ndarray:
    output = []
    for start in range(0, len(rows), 4):
        selected = rows[start : start + 4]
        value = torch.as_tensor(np.asarray(channel[selected]).copy(), device=DEVICE)
        descriptor = phase89.full_pas(value)
        output.append(descriptor.cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def build_anchor_targets(
    pos: np.ndarray,
    channel: np.ndarray,
    energy: np.ndarray,
    test_pos: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_labels = official_train_labels(pos, energy, test_pos, test_labels)
    anchor_rows = np.flatnonzero(np.isin(train_labels, ACTIVE_GROUPS))
    active_test_rows = np.flatnonzero(np.isin(test_labels, ACTIVE_GROUPS))
    anchor_labels = train_labels[anchor_rows]
    expected_anchor_counts = {5: 22, 6: 42}
    actual_anchor_counts = {
        group: int(np.sum(anchor_labels == group)) for group in ACTIVE_GROUPS
    }
    if actual_anchor_counts != expected_anchor_counts:
        raise RuntimeError(f"Active anchor-count invariant failed: {actual_anchor_counts}")
    if len(active_test_rows) != 89:
        raise RuntimeError(f"Active test-row invariant failed: {len(active_test_rows)}")

    anchor_truth = full_pas_descriptors(channel, anchor_rows)
    combined_pos = np.vstack((pos[anchor_rows], test_pos))
    combined_labels = np.concatenate((anchor_labels, test_labels))
    combined_shifts = horizontal_shifts(combined_pos)
    anchor_local = np.arange(len(anchor_rows), dtype=np.int64)
    query_local = len(anchor_rows) + active_test_rows
    target = anchor_prediction(
        combined_pos[:, :2],
        combined_labels,
        combined_shifts,
        anchor_local,
        query_local,
        anchor_truth,
        "horizontal",
    )
    if target.shape != (89, 256, 4, 192) or target.dtype != np.float32:
        raise RuntimeError(f"Unexpected anchor target: {target.shape} {target.dtype}")
    if not np.isfinite(target).all() or np.any(
        np.linalg.vector_norm(target, axis=1) <= 0
    ):
        raise RuntimeError("Invalid active anchor target")
    lookup = np.full(len(test_pos), -1, dtype=np.int64)
    lookup[active_test_rows] = np.arange(len(active_test_rows))
    diagnostics = {
        "active_groups": list(ACTIVE_GROUPS),
        "active_test_rows": int(len(active_test_rows)),
        "anchor_counts": {str(key): value for key, value in actual_anchor_counts.items()},
        "anchor_training_indices_sha256": array_sha256(anchor_rows),
        "anchor_truth_full192_sha256": array_sha256(anchor_truth),
        "anchor_target_full192_sha256": array_sha256(target),
        "alignment": "horizontal",
        "neighbors": 4,
        "distance_power": 2.0,
    }
    return target, lookup, diagnostics


def prediction_only_scales(p9: torch.Tensor) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pas = torch.abs(rp.bs_fft_torch(p9)) ** 2
    pas_pnorm = (
        torch.linalg.vector_norm(pas, dim=1).detach().cpu().numpy().astype(np.float32)
    )
    pdp = torch.abs(torch.fft.fft(p9, dim=-1, norm="ortho")) ** 2
    pdp_pnorm = (
        torch.linalg.vector_norm(pdp, dim=-1).detach().cpu().numpy().astype(np.float32)
    )
    return phase90.prediction_only_scales(
        {"pas_pnorm": pas_pnorm, "pdp_pnorm": pdp_pnorm}
    )


def quantiles(value: np.ndarray) -> dict[str, float]:
    flat = value.reshape(-1)
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


@torch.no_grad()
def build() -> dict[str, Any]:
    if OUTPUT.exists() or BUILDING.exists() or MANIFEST.exists():
        raise RuntimeError("Phase93 output/building/manifest exists; refusing to overwrite")
    source_audit = audit_sources()
    p9_file = validate_source_array(P9)
    p10_file = validate_source_array(P10)
    phase40_file = validate_source_array(PHASE40)

    pos, train_channel, energy = rp.load_data()
    if int(np.sum(energy <= 0)) != 262:
        raise RuntimeError("Zero-outlier invariant failed")
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    group_counts = {
        int(group): int(count)
        for group, count in zip(*np.unique(test_labels, return_counts=True))
    }
    if group_counts != {0: 90, 1: 21, 2: 2, 3: 63, 4: 67, 5: 43, 6: 46, 7: 50, 8: 22, 9: 55, 10: 41}:
        raise RuntimeError(f"Official group-count invariant failed: {group_counts}")
    anchor_target, active_lookup, anchor_diagnostics = build_anchor_targets(
        pos, train_channel, energy, test_pos, test_labels
    )
    print(json.dumps({"stage": "anchor_target", **anchor_diagnostics}), flush=True)

    output = np.lib.format.open_memmap(
        BUILDING, mode="w+", dtype=np.complex64, shape=EXPECTED_SHAPE
    )
    phase89_data_digest = hashlib.sha256()
    phase93_data_digest = hashlib.sha256()
    scale_rows = []
    q_pas_rows = []
    q_pdp_rows = []
    scale_one_branches_compared = 0
    scale_one_bitwise = True
    non_g56_rows_compared = 0
    non_g56_phase40_bitwise = True
    non_g56_phase40_max_abs_error = 0.0
    finite = True
    all_rows_nonzero = True

    for start in range(0, len(p9_file), BATCH):
        stop = min(start + BATCH, len(p9_file))
        batch_rows = np.arange(start, stop)
        x9 = torch.as_tensor(np.asarray(p9_file[start:stop]).copy(), device=DEVICE)
        x10 = torch.as_tensor(np.asarray(p10_file[start:stop]).copy(), device=DEVICE)
        g56 = x9.clone()
        active_np = np.isin(test_labels[start:stop], ACTIVE_GROUPS)
        if np.any(active_np):
            active = torch.as_tensor(active_np, device=DEVICE)
            base192 = phase89.full_pas(x9[active])
            target_indices = active_lookup[batch_rows[active_np]]
            retained = torch.as_tensor(
                anchor_target[target_indices].copy(), device=DEVICE
            )
            desired192 = phase89.normalize_dim1(
                (1.0 - DOSE) * base192 + DOSE * retained
            )
            g56[active] = project(x9[active], base192, desired192, 192, 12)

        direction = phase89.phase40_direction(x9, x10)
        phase89_value = phase89.apply_frozen_phase40(g56, direction)
        phase89_block = phase89_value.detach().cpu().numpy().astype(np.complex64)
        phase89_data_digest.update(memoryview(np.ascontiguousarray(phase89_block)).cast("B"))

        inactive_rows = np.flatnonzero(~active_np)
        if len(inactive_rows):
            reference = np.asarray(phase40_file[start:stop])[inactive_rows]
            actual = phase89_block[inactive_rows]
            non_g56_rows_compared += len(inactive_rows)
            non_g56_phase40_bitwise &= bool(np.array_equal(actual, reference))
            non_g56_phase40_max_abs_error = max(
                non_g56_phase40_max_abs_error,
                float(np.max(np.abs(actual - reference))),
            )

        scales, diagnostics = prediction_only_scales(x9)
        scale_rows.append(scales)
        q_pas_rows.append(diagnostics["q_pas"])
        q_pdp_rows.append(diagnostics["q_pdp"])
        phase93_value = phase89_value.clone()
        active_scale = scales > 1.0
        inactive_scale = ~active_scale
        for local_row, ue in np.argwhere(active_scale):
            phase93_value[int(local_row), :, int(ue), :] = (
                phase89_value[int(local_row), :, int(ue), :]
                * float(scales[local_row, ue])
            )
        phase93_block = phase93_value.detach().cpu().numpy().astype(np.complex64)
        for local_row, ue in np.argwhere(inactive_scale):
            scale_one_branches_compared += 1
            scale_one_bitwise &= bool(
                np.array_equal(
                    phase93_block[int(local_row), :, int(ue), :],
                    phase89_block[int(local_row), :, int(ue), :],
                )
            )
        finite &= bool(np.isfinite(phase93_block).all())
        all_rows_nonzero &= bool(
            np.all(np.sum(np.abs(phase93_block) ** 2, axis=(1, 2, 3)) > 0)
        )
        output[start:stop] = phase93_block
        phase93_data_digest.update(memoryview(np.ascontiguousarray(phase93_block)).cast("B"))
        if stop % 50 == 0 or stop == len(p9_file):
            output.flush()
            print(json.dumps({"stage": "build", "done": stop}), flush=True)

    output.flush()
    del output
    os.replace(BUILDING, OUTPUT)

    scales = np.concatenate(scale_rows)
    q_pas = np.concatenate(q_pas_rows)
    q_pdp = np.concatenate(q_pdp_rows)
    active_scale_branches = int(np.sum(scales > 1.0))
    inactive_scale_branches = int(np.sum(scales == 1.0))
    if active_scale_branches + inactive_scale_branches != 2000:
        raise RuntimeError("Scale branch count invariant failed")

    check = np.load(OUTPUT, mmap_mode="r")
    reload_finite = True
    reload_nonzero = True
    for start in range(0, len(check), 4):
        block = np.asarray(check[start : start + 4])
        reload_finite &= bool(np.isfinite(block).all())
        reload_nonzero &= bool(
            np.all(np.sum(np.abs(block) ** 2, axis=(1, 2, 3)) > 0)
        )
    qa = {
        "shape": list(check.shape),
        "dtype": str(check.dtype),
        "bytes": OUTPUT.stat().st_size,
        "finite_stream": finite,
        "finite_reload": reload_finite,
        "all_rows_nonzero_stream": all_rows_nonzero,
        "all_rows_nonzero_reload": reload_nonzero,
        "non_g56_rows_compared_to_phase40": non_g56_rows_compared,
        "non_g56_phase89_bitwise_phase40": non_g56_phase40_bitwise,
        "non_g56_phase89_vs_phase40_max_abs_error": non_g56_phase40_max_abs_error,
        "scale_one_branches_compared_to_phase89": scale_one_branches_compared,
        "scale_one_branches_bitwise_phase89": scale_one_bitwise,
        "active_scale_branches": active_scale_branches,
        "inactive_scale_branches": inactive_scale_branches,
        "scale_quantiles": quantiles(scales),
        "q_pas_quantiles": quantiles(q_pas),
        "q_pdp_quantiles": quantiles(q_pdp),
        "phase89_stream_data_sha256": phase89_data_digest.hexdigest(),
        "phase93_stream_data_sha256": phase93_data_digest.hexdigest(),
    }
    valid = bool(
        check.shape == EXPECTED_SHAPE
        and check.dtype == EXPECTED_DTYPE
        and OUTPUT.stat().st_size == EXPECTED_BYTES
        and finite
        and reload_finite
        and all_rows_nonzero
        and reload_nonzero
        and non_g56_phase40_bitwise
        and non_g56_phase40_max_abs_error == 0.0
        and scale_one_bitwise
        and scale_one_branches_compared == inactive_scale_branches
    )
    if not valid:
        raise RuntimeError(f"Phase93 QA failed: {qa}")

    payload = {
        "status": "complete",
        "protocol": "exact Phase89 test channel then original-P9-only Phase90 symmetric clamp-floor per UE",
        "builder": Path(__file__).name,
        "builder_sha256_before_manifest": sha256(Path(__file__).resolve()),
        "source_audit": source_audit,
        "zero_outliers_removed": int(np.sum(energy <= 0)),
        "official_group_counts": {str(key): value for key, value in group_counts.items()},
        "anchor_diagnostics": anchor_diagnostics,
        "frozen_parameters": {
            "phase79_active_groups": list(ACTIVE_GROUPS),
            "phase79_full192_dose": DOSE,
            "phase79_iterations": 12,
            "phase40_eta": phase89.ANTI_ETA,
            "phase90_low_tail_quantile": phase90.LOW_TAIL_QUANTILE,
            "phase90_symmetric_boundary": phase90.SYMMETRIC_BOUNDARY,
            "phase90_scale_bounds": list(phase90.SCALE_BOUNDS),
            "phase90_scale_source": "original P9 prediction only",
        },
        "output": {
            "path": str(OUTPUT),
            "name": OUTPUT.name,
            "sha256": sha256(OUTPUT),
            "shape": list(check.shape),
            "dtype": str(check.dtype),
            "bytes": OUTPUT.stat().st_size,
        },
        "qa": qa,
        "validation_basis": {
            "phase93_local_mean_delta_vs_phase40": 0.000806178657385237,
            "phase93_lcb_mean_minus_0.75std": 0.000271591632818545,
            "phase93_positive_folds": 4,
            "phase93_minimum_fold_delta": -0.000129698287416824,
        },
        "uploaded": False,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "complete", "output": payload["output"], "qa": qa}), flush=True)
    return payload


if __name__ == "__main__":
    build()
