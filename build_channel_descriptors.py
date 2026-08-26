from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent


def run() -> None:
    _, channel, energy = rp.load_data()
    pas = np.empty((len(channel), 256), np.float32)
    pdp = np.empty((len(channel), 192), np.float32)
    device = torch.device("cuda")
    for start in range(0, len(channel), 8):
        stop = min(start + 8, len(channel))
        h = torch.as_tensor(np.asarray(channel[start:stop]).copy(), device=device)
        p = torch.abs(rp.bs_fft_torch(h)) ** 2
        p = p.sum((2, 3))
        p /= torch.linalg.vector_norm(p, dim=1, keepdim=True).clamp_min(1e-30)
        d = torch.abs(torch.fft.fft(h, dim=-1, norm="ortho")) ** 2
        d = d.sum((1, 2))
        d /= torch.linalg.vector_norm(d, dim=1, keepdim=True).clamp_min(1e-30)
        pas[start:stop] = p.cpu().numpy()
        pdp[start:stop] = d.cpu().numpy()
        if stop % 400 == 0 or stop == len(channel):
            print(f"descriptors {stop}/{len(channel)}", flush=True)
    np.savez(
        ROOT / "channel_descriptors.npz",
        pas=pas,
        pdp=pdp,
        log_energy=np.log1p(energy).astype(np.float32),
        valid=(energy > 0),
    )


if __name__ == "__main__":
    run()
