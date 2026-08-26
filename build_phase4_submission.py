from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp
from matched_local_spectral_calibration import (
    apply_per_sample,
    predict_profiles,
    residual_profiles,
)
from matched_spectral_calibration import (
    DEVICE,
    apply_correction,
    ratio_profile,
    spectral_sums,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "Round2_Test_Channel_matched_phase4_base.npy"
AFFINE = ROOT / "Round2_Test_Channel_matched_phase4_g10_affine.npy"
GRADIENT = ROOT / "Round2_Test_Channel_matched_phase4_g10_gradient.npy"
OUTPUT = ROOT / "Round2_Test_Channel_matched_phase4_delta1239.npy"
GLOBAL_CONFIGS = {
    4: ("global", 0.50, 0.00, "pas_pdp"),
    5: ("global", 0.25, 0.00, "pas_pdp"),
    6: ("global", 0.25, 0.25, "pas_pdp"),
    7: ("global", 0.25, 0.25, "pas_pdp"),
    8: ("ue", 0.25, 0.00, "pas_pdp"),
}
LOCAL_CONFIGS = {0: (20, 0.20), 3: (10, 0.20)}


def align(candidate: np.ndarray, base: np.ndarray) -> np.ndarray:
    cross = np.sum(np.conj(candidate) * base, axis=(1, 2, 3), dtype=np.complex128)
    phase = cross / np.maximum(np.abs(cross), 1e-30)
    return candidate * phase[:, None, None, None].astype(np.complex64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run() -> None:
    pos, channel, _ = rp.load_data()
    query = np.load(ROOT / "Round2_Test_Pos.npy")
    query_groups = rp.official_island_labels(query)
    base = np.load(BASE, mmap_mode="r")
    affine = np.load(AFFINE, mmap_mode="r")
    gradient = np.load(GRADIENT, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), 4):
        stop = min(start + 4, len(base))
        output[start:stop] = base[start:stop]

    rows10 = np.flatnonzero(query_groups == 10)
    b10 = np.asarray(base[rows10])
    a10 = align(np.asarray(affine[rows10]), b10)
    g10 = align(np.asarray(gradient[rows10]), b10)
    output[rows10] = (b10 + (a10 - b10) + 0.10 * (g10 - b10)).astype(np.complex64)
    output.flush()
    print(json.dumps({"stage": "g10_combo"}), flush=True)

    folds = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        pred = np.load(ROOT / f"matched_phase2_full_fold{fold}.npy", mmap_mode="r")
        folds.append({
            "fold": fold,
            "val": val,
            "labels": labels,
            "pred": pred,
            "target": channel[val],
            "xy": pos[val, :2],
        })

    for group, (level, pas_blend, pdp_blend, order) in GLOBAL_CONFIGS.items():
        train_sums = []
        for row in folds:
            mask = row["labels"] == group
            train_sums.append(spectral_sums(row["pred"][mask], row["target"][mask]))
        pas_ratio = ratio_profile(train_sums, "pas", level)
        pdp_ratio = ratio_profile(train_sums, "pdp", level)
        rows = np.flatnonzero(query_groups == group)
        tensor = torch.as_tensor(np.asarray(output[rows]).copy(), device=DEVICE)
        output[rows] = apply_correction(
            tensor, pas_ratio, pdp_ratio, pas_blend, pdp_blend, order
        ).cpu().numpy().astype(np.complex64)
        output.flush()
        print(json.dumps({"stage": "global", "group": group}), flush=True)

    for group, (neighbors, blend) in LOCAL_CONFIGS.items():
        train_xy = []
        train_pas = []
        train_pdp = []
        for row in folds:
            mask = row["labels"] == group
            pas, pdp = residual_profiles(row["pred"][mask], row["target"][mask])
            train_xy.append(row["xy"][mask])
            train_pas.append(pas)
            train_pdp.append(pdp)
        rows = np.flatnonzero(query_groups == group)
        pas_ratio, pdp_ratio = predict_profiles(
            np.concatenate(train_xy),
            np.concatenate(train_pas),
            np.concatenate(train_pdp),
            query[rows, :2],
            neighbors,
        )
        tensor = torch.as_tensor(np.asarray(output[rows]).copy(), device=DEVICE)
        output[rows] = apply_per_sample(
            tensor, pas_ratio, pdp_ratio, blend
        ).cpu().numpy().astype(np.complex64)
        output.flush()
        print(json.dumps({"stage": "local", "group": group}), flush=True)

    del output
    check = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(check), 8):
        finite &= bool(np.isfinite(check[start:start + 8]).all())
    validation = json.loads(
        (ROOT / "matched_phase4_composite_validation.json").read_text(encoding="utf-8")
    )["summary"]
    manifest = {
        "output": OUTPUT.name,
        "shape": list(check.shape),
        "dtype": str(check.dtype),
        "bytes": OUTPUT.stat().st_size,
        "finite": finite,
        "sha256": sha256(OUTPUT),
        "validation": validation,
        "profiles": {
            "global": {str(group): value for group, value in GLOBAL_CONFIGS.items()},
            "local": {str(group): value for group, value in LOCAL_CONFIGS.items()},
            "g10_combo": {"affine_weight": 1.0, "gradient_weight": 0.10},
        },
    }
    (ROOT / "matched_phase4_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    run()
