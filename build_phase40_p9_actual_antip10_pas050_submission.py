from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
TEMP = Path(os.environ["TEMP"])
P9 = ROOT / "Round2_Test_Channel_matched_phase9_buildable_anchor_joint.npy"
P10 = ROOT / "Round2_Test_Channel_phase10_core125_complement100_anchor_pas.npy"
OUTPUT = TEMP / "Round2_Test_Channel_phase40_p9_actual_antip10_pas050.npy"
MANIFEST = ROOT / "phase40_p9_actual_antip10_pas050_manifest.json"
ETA = 0.50
EPSILON = 1e-4 / 256
BATCH = 2
DEVICE = torch.device("cuda")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=1, keepdim=True).clamp_min(1e-30)


def band24(value: torch.Tensor) -> torch.Tensor:
    power = torch.abs(rp.bs_fft_torch(value)) ** 2
    power = normalize(power)
    return normalize(power.reshape(len(value), 256, 4, 24, 8).mean(-1))


@torch.no_grad()
def build() -> None:
    if OUTPUT.exists() or MANIFEST.exists():
        raise RuntimeError("Phase40 output already exists; refusing to overwrite")
    for path in (P9, P10):
        value = np.load(path, mmap_mode="r")
        if value.shape != (500, 256, 4, 192) or value.dtype != np.complex64:
            raise RuntimeError(f"unexpected source {path.name}: {value.shape} {value.dtype}")
    p9 = np.load(P9, mmap_mode="r")
    p10 = np.load(P10, mmap_mode="r")
    building = OUTPUT.with_name(OUTPUT.name + ".building.npy")
    if building.exists():
        raise RuntimeError(f"stale building file exists: {building}")
    out = np.lib.format.open_memmap(
        building, mode="w+", dtype=np.complex64, shape=p9.shape
    )
    clip_count = 0
    total_count = 0
    log_rms = []
    for start in range(0, len(p9), BATCH):
        stop = min(start + BATCH, len(p9))
        x9 = torch.as_tensor(np.asarray(p9[start:stop]).copy(), device=DEVICE)
        x10 = torch.as_tensor(np.asarray(p10[start:stop]).copy(), device=DEVICE)
        b9, b10 = band24(x9), band24(x10)
        direction = torch.log((b10 + EPSILON) / (b9 + EPSILON)).clamp(-2.0, 2.0)
        desired = normalize(b9 * torch.exp(-ETA * direction))
        spectrum = rp.bs_fft_torch(x9)
        power = torch.abs(spectrum) ** 2
        ratio = (desired + EPSILON) / (b9 + EPSILON)
        clip_count += int(torch.sum((ratio < 0.25) | (ratio > 4.0)).cpu())
        total_count += ratio.numel()
        log_rms.append(float(torch.sqrt(torch.mean(direction * direction)).cpu()))
        target = power * ratio.repeat_interleave(8, dim=3)
        candidate = rp.bs_ifft_torch(
            spectrum * torch.sqrt(target / power.clamp_min(1e-30))
        )
        block = candidate.detach().cpu().numpy().astype(np.complex64)
        if not np.isfinite(block).all() or np.any(np.sum(np.abs(block) ** 2, axis=(1, 2, 3)) <= 0):
            raise RuntimeError(f"invalid output rows {start}:{stop}")
        out[start:stop] = block
    out.flush()
    del out
    os.replace(building, OUTPUT)

    reloaded = np.load(OUTPUT, mmap_mode="r")
    finite = True
    nonzero = True
    nmse_num = 0.0
    nmse_den = 0.0
    for start in range(0, len(reloaded), 4):
        stop = min(start + 4, len(reloaded))
        block = np.asarray(reloaded[start:stop])
        base = np.asarray(p9[start:stop])
        finite &= bool(np.isfinite(block).all())
        nonzero &= bool(np.all(np.sum(np.abs(block) ** 2, axis=(1, 2, 3)) > 0))
        nmse_num += float(np.sum(np.abs(block - base) ** 2, dtype=np.float64))
        nmse_den += float(np.sum(np.abs(base) ** 2, dtype=np.float64))
    payload = {
        "status": "complete",
        "protocol": "official-feedback anti-P10 actual realized 24-band PAS direct projection",
        "eta": ETA,
        "sources": {"p9": {"path": str(P9), "sha256": sha256_file(P9)},
                    "p10": {"path": str(P10), "sha256": sha256_file(P10)}},
        "output": {"path": str(OUTPUT), "sha256": sha256_file(OUTPUT),
                   "shape": list(reloaded.shape), "dtype": str(reloaded.dtype),
                   "finite": finite, "all_rows_nonzero": nonzero},
        "qa": {"mean_bad_direction_log_rms": float(np.mean(log_rms)),
               "ratio_clip_fraction": clip_count / total_count,
               "nmse_vs_p9": nmse_num / max(nmse_den, 1e-30)},
        "official_feedback_basis": {"p9": 0.6395, "p10": 0.6354,
                                    "scale050": 0.638513, "scale075": 0.6376,
                                    "scale100": 0.636432},
        "local_exact_warning": "anti direction is locally negative; this is an official-feedback probe, not a local-score milestone",
        "uploaded": False,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        print(json.dumps({"status": "plan", "output": str(OUTPUT), "eta": ETA}))
        return
    build()


if __name__ == "__main__":
    main()
