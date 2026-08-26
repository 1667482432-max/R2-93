from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BANDS = 24
BAND_WIDTH = 8
SCALE = 1.25
PDP_STRENGTH = 1.5
ITERATIONS = 12


def normalized(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=dim, keepdim=True).clamp_min(1e-30)


@torch.no_grad()
def run() -> None:
    for fold in range(5):
        output_path = ROOT / f"matched_phase6_full_fold{fold}.npy"
        if output_path.exists():
            print(json.dumps({"stage": "exists", "fold": fold}), flush=True)
            continue
        base = np.load(ROOT / f"matched_phase5_full_fold{fold}.npy", mmap_mode="r")
        desired = np.load(
            ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
            mmap_mode="r",
        )
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.complex64, shape=base.shape
        )
        for start in range(0, len(base), 4):
            stop = min(start + 4, len(base))
            p = torch.as_tensor(np.asarray(base[start:stop]).copy(), device=DEVICE)
            base_pas_complex = rp.bs_fft_torch(p)
            base_pas = torch.abs(base_pas_complex) ** 2
            base_band = normalized(
                normalized(base_pas, 1)
                .reshape(len(p), 256, 4, BANDS, BAND_WIDTH)
                .mean(4),
                1,
            )
            desired_band = torch.as_tensor(
                np.asarray(desired[start:stop]).copy(), device=DEVICE
            )
            target_band = normalized(
                (1.0 - SCALE) * base_band + SCALE * desired_band, 1
            )
            epsilon = 1e-3 / base_band.shape[1]
            ratio = ((target_band + epsilon) / (base_band + epsilon)).clamp(0.25, 4.0)
            target_pas = base_pas * ratio.repeat_interleave(BAND_WIDTH, dim=3)
            base_pdp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
            x = rp.bs_ifft_torch(
                base_pas_complex * torch.sqrt(target_pas / base_pas.clamp_min(1e-30))
            )
            for _ in range(ITERATIONS):
                z = torch.fft.fft(x, dim=-1, norm="ortho")
                correction = (
                    torch.sqrt(base_pdp).clamp_min(1e-30)
                    / torch.abs(z).clamp_min(1e-30)
                )
                x = torch.fft.ifft(
                    z * correction.pow(PDP_STRENGTH), dim=-1, norm="ortho"
                )
                z = rp.bs_fft_torch(x)
                correction = (
                    torch.sqrt(target_pas).clamp_min(1e-30)
                    / torch.abs(z).clamp_min(1e-30)
                )
                x = rp.bs_ifft_torch(z * correction)
            output[start:stop] = x.cpu().numpy().astype(np.complex64)
            if stop % 64 == 0 or stop == len(base):
                print(
                    json.dumps(
                        {"stage": "rows", "fold": fold, "done": stop, "total": len(base)}
                    ),
                    flush=True,
                )
        output.flush()
        del output
        print(
            json.dumps(
                {"stage": "fold", "fold": fold, "bytes": output_path.stat().st_size}
            ),
            flush=True,
        )


if __name__ == "__main__":
    run()
