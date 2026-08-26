from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA

import r2_pipeline as rp
os.environ.setdefault("R2_BANDS", "24")
from matched_local_spectral_calibration import apply_per_sample
from matched_phase5_composite_validation import SAFE, prepare_folds
from matched_phase5_feature_residual import feature_matrix, predict_ratios, prediction_summaries
from matched_phase5_tree_band_descriptor import (
    base_descriptor,
    build_cache,
    fit_predict,
    normalize_last,
)
from matched_phase5_tree_descriptor import model_features


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BANDS = 24
BAND_WIDTH = 192 // BANDS
BASE = ROOT / "Round2_Test_Channel_matched_phase4_delta1239.npy"
TREE_TEST = ROOT / f"matched_tree_pas_band{BANDS}_test.npy"
RBF_TEST = ROOT / f"matched_rbf_pas_band{BANDS}_test.npy"
OUTPUT = ROOT / "Round2_Test_Channel_matched_phase5_delta1642.npy"
SCALE = 1.0
PDP_STRENGTH = 1.5
ITERATIONS = 24


def normalized(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=dim, keepdim=True).clamp_min(1e-30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def train_descriptors() -> tuple[np.ndarray, np.ndarray]:
    pos, _, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    all_pos = np.vstack((pos, test_pos))
    map_features = np.load(ROOT / "los_map_features.npy")
    features = model_features(all_pos, map_features)
    target = build_cache()
    valid = np.flatnonzero(energy > 0)
    test_indices = np.arange(len(pos), len(all_pos))

    if TREE_TEST.exists():
        tree_prediction = np.load(TREE_TEST, mmap_mode="r")
    else:
        tree_prediction = fit_predict(
            features, target, valid, test_indices, seed=14567
        )
        np.save(TREE_TEST, tree_prediction)
    print(json.dumps({"stage": "tree_test", "shape": list(tree_prediction.shape)}), flush=True)

    if RBF_TEST.exists():
        rbf_prediction = np.load(RBF_TEST, mmap_mode="r")
    else:
        shaped = np.asarray(target[valid]).transpose(0, 2, 3, 1)
        transformed = np.sqrt(np.maximum(shaped, 0)).reshape(len(valid), -1)
        pca = PCA(n_components=160, svd_solver="randomized", random_state=20100)
        coefficient = pca.fit_transform(transformed)
        scale = np.std(pos[valid, :2], axis=0)
        model = RBFInterpolator(
            pos[valid, :2] / scale,
            coefficient,
            neighbors=100,
            smoothing=0.1,
            kernel="linear",
            degree=0,
        )
        predicted = pca.inverse_transform(model(test_pos[:, :2] / scale))
        predicted = np.maximum(predicted.reshape(len(test_pos), 4, BANDS, 256), 0) ** 2
        rbf_prediction = normalize_last(predicted).transpose(0, 3, 1, 2).astype(np.float32)
        np.save(RBF_TEST, rbf_prediction)
    print(json.dumps({"stage": "rbf_test", "shape": list(rbf_prediction.shape)}), flush=True)
    return tree_prediction, rbf_prediction


def test_residual_ratios(base: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    pos, _, _ = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    groups = rp.official_island_labels(test_pos)
    map_features = np.load(ROOT / "los_map_features.npy")
    folds = prepare_folds()
    query = {
        "xy": test_pos[:, :2],
        "map": map_features[len(pos):],
        "spec": prediction_summaries(base),
    }
    output = {}
    for group, (feature_name, neighbors, _, axis) in SAFE.items():
        train_masks = [row["labels"] == group for row in folds]
        query_mask = groups == group
        train_feature, query_feature = feature_matrix(
            feature_name, folds, query, train_masks, query_mask
        )
        train_pas = np.concatenate(
            [row["pas"][mask] for row, mask in zip(folds, train_masks)]
        )
        train_pdp = np.concatenate(
            [row["pdp"][mask] for row, mask in zip(folds, train_masks)]
        )
        pas, pdp = predict_ratios(
            train_feature, query_feature, train_pas, train_pdp, neighbors
        )
        if axis == "pdp":
            pas = np.ones_like(pas)
        output[group] = pas, pdp
    return groups, output


@torch.no_grad()
def generate(tree_prediction: np.ndarray, rbf_prediction: np.ndarray) -> None:
    base = np.load(BASE, mmap_mode="r")
    base_band = base_descriptor(base)
    groups, safe_ratios = test_residual_ratios(base)
    group_offsets = {
        group: np.cumsum(np.r_[0, groups == group])[:-1] for group in SAFE
    }
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), 4):
        stop = min(start + 4, len(base))
        p = torch.as_tensor(np.asarray(base[start:stop]).copy(), device=DEVICE)
        base_pas_complex = rp.bs_fft_torch(p)
        base_pas = torch.abs(base_pas_complex) ** 2
        band = torch.as_tensor(np.asarray(base_band[start:stop]).copy(), device=DEVICE)
        tree = torch.as_tensor(np.asarray(tree_prediction[start:stop]).copy(), device=DEVICE)
        rbf = torch.as_tensor(np.asarray(rbf_prediction[start:stop]).copy(), device=DEVICE)
        desired = normalized(0.60 * band + 0.25 * tree + 0.15 * rbf, 1)
        epsilon = 1e-3 / band.shape[1]
        ratio = ((desired + epsilon) / (band + epsilon)).clamp(0.25, 4.0)
        target_pas = base_pas * ratio.repeat_interleave(BAND_WIDTH, dim=3)
        base_pdp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
        x = rp.bs_ifft_torch(
            base_pas_complex * torch.sqrt(target_pas / base_pas.clamp_min(1e-30))
        )
        for _ in range(ITERATIONS):
            z = torch.fft.fft(x, dim=-1, norm="ortho")
            correction = torch.sqrt(base_pdp).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
            x = torch.fft.ifft(z * correction.pow(PDP_STRENGTH), dim=-1, norm="ortho")
            z = rp.bs_fft_torch(x)
            correction = torch.sqrt(target_pas).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
            x = rp.bs_ifft_torch(z * correction)
        for group in SAFE:
            local = np.flatnonzero(groups[start:stop] == group)
            if not len(local):
                continue
            indices = group_offsets[group][start + local]
            pas_ratio, pdp_ratio = safe_ratios[group]
            local_tensor = torch.as_tensor(local, device=DEVICE)
            x[local_tensor] = apply_per_sample(
                x[local_tensor],
                pas_ratio[indices],
                pdp_ratio[indices],
                SAFE[group][2],
            )
        output[start:stop] = x.cpu().numpy().astype(np.complex64)
        if stop % 50 == 0 or stop == len(base):
            output.flush()
            print(json.dumps({"stage": "generate", "done": stop}), flush=True)
    del output


def validate_output() -> dict:
    check = np.load(OUTPUT, mmap_mode="r")
    finite = True
    nonzero = True
    for start in range(0, len(check), 8):
        block = np.asarray(check[start:start + 8])
        finite &= bool(np.isfinite(block).all())
        nonzero &= bool(np.all(np.sum(np.abs(block), axis=(1, 2, 3)) > 0))
    validation = json.loads(
        (ROOT / "matched_phase5_composite_validation.json").read_text(encoding="utf-8")
    )["summary"]
    selected = next(
        row
        for row in validation
        if row["projection"] == [1.0, 1.5, 24]
        and row["groups"] == [0, 1, 5]
        and row["pdp_tree_alpha"] == 0.0
    )
    manifest = {
        "output": OUTPUT.name,
        "shape": list(check.shape),
        "dtype": str(check.dtype),
        "bytes": OUTPUT.stat().st_size,
        "finite": finite,
        "all_test_rows_nonzero": nonzero,
        "sha256": sha256(OUTPUT),
        "validation": selected,
        "delta_vs_phase4": selected["mean_delta"],
        "cumulative_delta_vs_v3": 0.01239 + selected["mean_delta"],
        "model": {
            "bands": BANDS,
            "tree_weight": 0.25,
            "rbf_weight": 0.15,
            "base_weight": 0.60,
            "projection_scale": SCALE,
            "pdp_strength": PDP_STRENGTH,
            "iterations": ITERATIONS,
            "safe_groups": SAFE,
            "zero_channel_training_rows_removed": True,
        },
    }
    (ROOT / "matched_phase5_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)
    return manifest


def run() -> None:
    tree_prediction, rbf_prediction = train_descriptors()
    generate(tree_prediction, rbf_prediction)
    validate_output()


if __name__ == "__main__":
    run()
