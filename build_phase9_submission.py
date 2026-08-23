from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import base_descriptor
from phase8_anchor_augmented_local_pas_screen import local_prediction
from phase8_anchor_local_gate_channel_validation import ALPHA_GRID, prepare_folds
from phase8_anchor_retained_pas_screen import horizontal_shifts, normalize
from phase9_anchor_joint_channel_validation import normalize_last, project_joint
from phase9_buildable_full_pdp_screen import descriptor as pdp_descriptor


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BASE = ROOT / "Round2_Test_Channel_matched_phase6_delta2053.npy"
OUTPUT = ROOT / "Round2_Test_Channel_matched_phase9_buildable_anchor_joint.npy"
MANIFEST = ROOT / "matched_phase9_buildable_anchor_joint_manifest.json"
PAS_RESIDUAL_ALPHA = 0.15
LOCAL_SCALE = 0.75
PDP_ALPHA = 0.025
ITERATIONS = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def log(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}), flush=True)


def official_anchors(
    pos: np.ndarray,
    energy: np.ndarray,
    test_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    test_labels = rp.official_island_labels(test_pos).astype(np.int64)
    groups = sorted(int(group) for group in np.unique(test_labels))
    boxes = {
        group: (
            test_pos[test_labels == group, :2].min(0),
            test_pos[test_labels == group, :2].max(0),
        )
        for group in groups
    }
    anchor_indices = []
    anchor_labels = []
    for index in np.flatnonzero(energy > 0):
        for group in groups:
            lo, hi = boxes[group]
            if np.all(pos[index, :2] >= lo) and np.all(pos[index, :2] <= hi):
                anchor_indices.append(int(index))
                anchor_labels.append(group)
                break
    anchors = np.asarray(anchor_indices, dtype=np.int64)
    labels = np.asarray(anchor_labels, dtype=np.int64)
    counts = {group: int(np.sum(labels == group)) for group in groups}
    return anchors, labels, test_labels, counts


def gate_features_test(
    pos: np.ndarray,
    test_pos: np.ndarray,
    valid: np.ndarray,
    anchors: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray,
    local: np.ndarray,
) -> np.ndarray:
    distance, neighbor_local = cKDTree(pos[valid, :2]).query(test_pos[:, :2], k=8)
    neighbors = valid[neighbor_local]
    inner = np.isin(neighbors, anchors).astype(np.float32)
    numerator = np.sum(base * local, axis=1)
    denominator = np.linalg.norm(base, axis=1) * np.linalg.norm(local, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-30)
    relative = np.zeros((len(test_pos), 4), dtype=np.float32)
    for group in np.unique(labels):
        rows = np.flatnonzero(labels == group)
        xy = test_pos[rows, :2]
        lo, hi = xy.min(0), xy.max(0)
        relative[rows, :2] = (xy - lo) / np.maximum(hi - lo, 1.0)
        relative[rows, 2:] = (xy - xy.mean(0)) / np.maximum(xy.std(0), 1.0)
    xy = test_pos[:, :2]
    side = xy[:, 1] > 0
    bs = np.where(side[:, None], np.array([52.0, 35.0]), np.array([-18.413, -65.881]))
    delta = xy - bs
    radius = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = delta / np.maximum(radius, 1e-9)
    one_hot = np.eye(11, dtype=np.float32)[labels]
    return np.column_stack(
        [
            distance, inner.sum(1), inner[:, :4].sum(1),
            cosine.mean((1, 2)), cosine.std((1, 2)), cosine.min((1, 2)),
            np.quantile(cosine, [0.1, 0.5, 0.9], axis=(1, 2)).T,
            relative, radius, unit, one_hot,
        ]
    ).astype(np.float32)


def interpolate_pas_residual(
    pos: np.ndarray,
    test_pos: np.ndarray,
    test_labels: np.ndarray,
    anchors: np.ndarray,
    anchor_labels: np.ndarray,
    anchor_log: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(test_pos), 256, 4, 24), dtype=np.float32)
    for group in np.unique(test_labels):
        rows = np.flatnonzero(test_labels == group)
        source = np.flatnonzero(anchor_labels == group)
        if len(source) == 0:
            continue
        k = min(16, len(source))
        distance, local = cKDTree(pos[anchors[source], :2]).query(test_pos[rows, :2], k=k)
        distance = np.asarray(distance)
        local = np.asarray(local)
        if k == 1:
            distance = distance[:, None]
            local = local[:, None]
        scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
        weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25)
        weight /= weight.sum(1, keepdims=True)
        output[rows] = np.einsum(
            "rk,rkaub->raub", weight, anchor_log[source[local]], optimize=True
        )
    return output


def interpolate_pdp_residual(
    pos: np.ndarray,
    test_pos: np.ndarray,
    test_labels: np.ndarray,
    anchors: np.ndarray,
    anchor_labels: np.ndarray,
    anchor_log: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(test_pos), 256, 4, 192), dtype=np.float32)
    for group in np.unique(test_labels):
        rows = np.flatnonzero(test_labels == group)
        source = np.flatnonzero(anchor_labels == group)
        if len(source) == 0:
            continue
        k = min(4, len(source))
        distance, local = cKDTree(pos[anchors[source], :2]).query(test_pos[rows, :2], k=k)
        distance = np.asarray(distance)
        local = np.asarray(local)
        if k == 1:
            distance = distance[:, None]
            local = local[:, None]
        scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
        weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** 0.5
        weight /= weight.sum(1, keepdims=True)
        output[rows] = np.einsum(
            "rk,rkaus->raus", weight, anchor_log[source[local]], optimize=True
        )
    return output


@torch.no_grad()
def generate() -> None:
    pos, channel, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    anchors, anchor_labels, test_labels, anchor_counts = official_anchors(pos, energy, test_pos)
    valid = np.flatnonzero(energy > 0)
    external_train = np.setdiff1d(valid, anchors)
    all_pos = np.vstack((pos, test_pos))
    test_index = np.arange(len(pos), len(pos) + len(test_pos))
    target_pas = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    shifts = horizontal_shifts(all_pos)
    base_channel = np.load(BASE, mmap_mode="r")
    base_pas = base_descriptor(base_channel).astype(np.float32)
    local_pas = local_prediction(
        all_pos[:, :2], shifts, target_pas, valid, test_index, 4, 3.0, "none"
    )

    anchor_pas_base = local_prediction(
        all_pos[:, :2], shifts, target_pas, external_train, anchors, 4, 3.0, "none"
    )
    epsilon_pas = 1e-4 / 256
    anchor_pas_log = np.clip(
        np.log((np.asarray(target_pas[anchors]) + epsilon_pas) / (anchor_pas_base + epsilon_pas)),
        -2.0, 2.0,
    )
    anchor_pas_log = np.repeat(
        anchor_pas_log.mean(3, keepdims=True), anchor_pas_log.shape[3], axis=3
    )
    pas_residual = interpolate_pas_residual(
        pos, test_pos, test_labels, anchors, anchor_labels, anchor_pas_log
    )

    folds, _, _, _, _, _ = prepare_folds()
    gate_x = np.concatenate([row["x"] for row in folds])
    gate_y = np.concatenate([row["gain_grid"] for row in folds])
    gate_w = np.concatenate([row["weights"] for row in folds])
    test_x = gate_features_test(
        pos, test_pos, valid, anchors, test_labels, base_pas, local_pas
    )
    gate = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=80, max_features=0.7,
        n_jobs=-1, random_state=52180,
    )
    gate.fit(gate_x, gate_y, sample_weight=gate_w)
    alpha = LOCAL_SCALE * ALPHA_GRID[np.argmax(gate.predict(test_x), axis=1)]
    alpha = np.clip(alpha, 0.0, 0.6).astype(np.float32)[:, None, None, None]
    corrected_pas = normalize(base_pas * np.exp(PAS_RESIDUAL_ALPHA * pas_residual))
    desired_pas = normalize((1.0 - alpha) * corrected_pas + alpha * local_pas).astype(np.float32)
    np.save(ROOT / "matched_phase9_buildable_pas_band24_test.npy", desired_pas)
    del folds, gate_x, gate_y, gate_w, test_x, gate

    distance, external_local = cKDTree(pos[external_train, :2]).query(pos[anchors, :2], k=8)
    external_indices = external_train[external_local]
    unique_indices, inverse = np.unique(external_indices, return_inverse=True)
    external_pdp = pdp_descriptor(channel, unique_indices)[inverse].reshape(
        len(anchors), 8, 256, 4, 192
    )
    scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
    weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** 3.0
    weight /= weight.sum(1, keepdims=True)
    anchor_pdp_base = np.einsum("nk,nkaus->naus", weight, external_pdp, optimize=True)
    anchor_pdp_truth = pdp_descriptor(channel, anchors)
    epsilon_pdp = 1e-4 / 192
    anchor_pdp_log = np.clip(
        np.log((anchor_pdp_truth + epsilon_pdp) / (anchor_pdp_base + epsilon_pdp)),
        -2.0, 2.0,
    )
    pdp_residual = interpolate_pdp_residual(
        pos, test_pos, test_labels, anchors, anchor_labels, anchor_pdp_log
    )
    base_pdp_path = ROOT / "matched_phase9_base_pdp_band192_test.npy"
    if base_pdp_path.exists():
        base_pdp = np.load(base_pdp_path, mmap_mode="r")
    else:
        output_pdp = np.lib.format.open_memmap(
            base_pdp_path, mode="w+", dtype=np.float32, shape=base_channel.shape
        )
        for start in range(0, len(base_channel), 4):
            x = torch.as_tensor(np.asarray(base_channel[start:start + 4]).copy(), device=DEVICE)
            pdp = torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2
            pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
            output_pdp[start:start + len(x)] = pdp.cpu().numpy().astype(np.float32)
        output_pdp.flush()
        del output_pdp
        base_pdp = np.load(base_pdp_path, mmap_mode="r")
    desired_pdp = normalize_last(
        np.asarray(base_pdp) * np.exp(PDP_ALPHA * pdp_residual)
    ).astype(np.float32)
    np.save(ROOT / "matched_phase9_buildable_pdp_band192_test.npy", desired_pdp)

    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base_channel.shape
    )
    for start in range(0, len(base_channel), 2):
        stop = min(start + 2, len(base_channel))
        p = torch.as_tensor(np.asarray(base_channel[start:stop]).copy(), device=DEVICE)
        value = project_joint(
            p,
            torch.as_tensor(base_pas[start:stop].copy(), device=DEVICE),
            torch.as_tensor(desired_pas[start:stop].copy(), device=DEVICE),
            torch.as_tensor(desired_pdp[start:stop].copy(), device=DEVICE),
            ITERATIONS,
        )
        output[start:stop] = value.cpu().numpy().astype(np.complex64)
        if stop % 50 == 0 or stop == len(base_channel):
            output.flush()
            log("generate", done=stop)
    output.flush()
    del output
    log(
        "generation_complete",
        anchors=int(len(anchors)), anchor_counts={str(k): v for k, v in anchor_counts.items()},
        gate_alpha_mean=float(alpha.mean()),
    )


def validate() -> dict:
    check = np.load(OUTPUT, mmap_mode="r")
    finite = True
    nonzero = True
    for start in range(0, len(check), 8):
        block = np.asarray(check[start:start + 8])
        finite &= bool(np.isfinite(block).all())
        nonzero &= bool(np.all(np.sum(np.abs(block), axis=(1, 2, 3)) > 0))
    validation = json.loads(
        (ROOT / "phase9_buildable_joint_channel_validation.json").read_text(encoding="utf-8")
    )["summary"]
    selected = next(
        row for row in validation
        if row["pas_label"] == "robust" and row["pdp_alpha"] == PDP_ALPHA
    )
    cumulative = 0.020525418923835668 + selected["mean_delta"]
    manifest = {
        "output": OUTPUT.name,
        "shape": list(check.shape), "dtype": str(check.dtype),
        "bytes": OUTPUT.stat().st_size, "finite": finite,
        "all_test_rows_nonzero": nonzero, "sha256": sha256(OUTPUT),
        "validation": selected,
        "delta_vs_phase6": selected["mean_delta"],
        "cumulative_delta_vs_matched_v3": cumulative,
        "training_rows_used": 3738,
        "zero_channel_outliers_removed": 262,
        "model": {
            "validation_split": "five official-like rectangles with matched internal-anchor retention",
            "anchor_baseline": "whole-rectangle-excluded external-neighbor prediction",
            "pas_residual_alpha": PAS_RESIDUAL_ALPHA,
            "local_gate_scale": LOCAL_SCALE,
            "pdp_residual_alpha": PDP_ALPHA,
            "joint_projection_iterations": ITERATIONS,
            "selection_policy": "fully buildable path; all five folds positive; robust LCB prioritized",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("validated", **manifest)
    if not finite or not nonzero:
        raise RuntimeError("submission contains non-finite or all-zero rows")
    return manifest


def run() -> None:
    generate()
    validate()


if __name__ == "__main__":
    run()
