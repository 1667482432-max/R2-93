from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
CONFIGS = (
    ("pas_ue025", "ue", 0.25, 0.0, "pas_pdp"),
    ("pas_ue050", "ue", 0.50, 0.0, "pas_pdp"),
    ("pas_ue075", "ue", 0.75, 0.0, "pas_pdp"),
    ("pas_ue100", "ue", 1.00, 0.0, "pas_pdp"),
    ("pdp_ue025", "ue", 0.0, 0.25, "pas_pdp"),
    ("pdp_ue050", "ue", 0.0, 0.50, "pas_pdp"),
    ("both_ue010", "ue", 0.10, 0.10, "pas_pdp"),
    ("both_ue025", "ue", 0.25, 0.25, "pas_pdp"),
    ("both_ue050", "ue", 0.50, 0.50, "pas_pdp"),
    ("both_ue075", "ue", 0.75, 0.75, "pas_pdp"),
    ("both_ue100", "ue", 1.00, 1.00, "pas_pdp"),
    ("both_ue025_reverse", "ue", 0.25, 0.25, "pdp_pas"),
    ("both_global025", "global", 0.25, 0.25, "pas_pdp"),
    ("both_global010", "global", 0.10, 0.10, "pas_pdp"),
    ("both_global040", "global", 0.40, 0.40, "pas_pdp"),
    ("both_global050", "global", 0.50, 0.50, "pas_pdp"),
    ("pas_global025", "global", 0.25, 0.0, "pas_pdp"),
    ("pas_global010", "global", 0.10, 0.0, "pas_pdp"),
    ("pas_global050", "global", 0.50, 0.0, "pas_pdp"),
    ("pas_global075", "global", 0.75, 0.0, "pas_pdp"),
)


def fold_data(channel, fold: int):
    val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
    labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
    pred = np.load(ROOT / f"matched_phase2_full_fold{fold}.npy", mmap_mode="r")
    return val, labels, pred, channel[val]


@torch.no_grad()
def spectral_sums(pred, target):
    sums = {
        "pas_pred": torch.zeros((256, 4), device=DEVICE, dtype=torch.float64),
        "pas_true": torch.zeros((256, 4), device=DEVICE, dtype=torch.float64),
        "pdp_pred": torch.zeros((4, 192), device=DEVICE, dtype=torch.float64),
        "pdp_true": torch.zeros((4, 192), device=DEVICE, dtype=torch.float64),
    }
    for start in range(0, len(pred), 4):
        stop = min(start + 4, len(pred))
        p = torch.as_tensor(np.asarray(pred[start:stop]).copy(), device=DEVICE)
        t = torch.as_tensor(np.asarray(target[start:stop]).copy(), device=DEVICE)
        pp = torch.abs(rp.bs_fft_torch(p)) ** 2
        tt = torch.abs(rp.bs_fft_torch(t)) ** 2
        pp /= torch.linalg.vector_norm(pp, dim=1, keepdim=True).clamp_min(1e-30)
        tt /= torch.linalg.vector_norm(tt, dim=1, keepdim=True).clamp_min(1e-30)
        sums["pas_pred"] += pp.sum((0, 3), dtype=torch.float64)
        sums["pas_true"] += tt.sum((0, 3), dtype=torch.float64)
        pp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
        tt = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
        pp /= torch.linalg.vector_norm(pp, dim=-1, keepdim=True).clamp_min(1e-30)
        tt /= torch.linalg.vector_norm(tt, dim=-1, keepdim=True).clamp_min(1e-30)
        sums["pdp_pred"] += pp.sum((0, 1), dtype=torch.float64)
        sums["pdp_true"] += tt.sum((0, 1), dtype=torch.float64)
    return {key: value.cpu().numpy() for key, value in sums.items()}


def ratio_profile(train_sums, axis: str, level: str):
    pred = sum(row[f"{axis}_pred"] for row in train_sums)
    true = sum(row[f"{axis}_true"] for row in train_sums)
    if level == "global":
        if axis == "pas":
            pred, true = pred.sum(1), true.sum(1)
        else:
            pred, true = pred.sum(0), true.sum(0)
    scale = np.maximum(np.mean(pred), 1e-30)
    ratio = (true + scale * 1e-3) / (pred + scale * 1e-3)
    ratio = np.clip(ratio, 0.25, 4.0)
    if ratio.ndim == 2:
        ratio /= np.exp(np.mean(np.log(ratio), axis=0, keepdims=True))
    else:
        ratio /= np.exp(np.mean(np.log(ratio)))
    return ratio.astype(np.float32)


@torch.no_grad()
def apply_correction(x, pas_ratio, pdp_ratio, pas_blend, pdp_blend, order):
    def pas_step(value):
        if pas_blend == 0:
            return value
        ratio = torch.as_tensor(pas_ratio, device=DEVICE)
        if ratio.ndim == 1:
            ratio = ratio[None, :, None, None]
        else:
            ratio = ratio[None, :, :, None]
        z = rp.bs_fft_torch(value)
        z *= ratio.pow(0.5 * pas_blend)
        return rp.bs_ifft_torch(z)

    def pdp_step(value):
        if pdp_blend == 0:
            return value
        ratio = torch.as_tensor(pdp_ratio, device=DEVICE)
        if ratio.ndim == 1:
            ratio = ratio[None, None, None, :]
        else:
            ratio = ratio[None, None, :, :]
        z = torch.fft.fft(value, dim=-1, norm="ortho")
        z *= ratio.pow(0.5 * pdp_blend)
        return torch.fft.ifft(z, dim=-1, norm="ortho")

    return pdp_step(pas_step(x)) if order == "pas_pdp" else pas_step(pdp_step(x))


@torch.no_grad()
def torch_score(pred, target):
    pas = pdp = error = energy = 0.0
    pas_n = pdp_n = 0
    for start in range(0, len(pred), 4):
        stop = min(start + 4, len(pred))
        p = pred[start:stop]
        t = target[start:stop]
        pp = torch.abs(rp.bs_fft_torch(p)) ** 2
        tt = torch.abs(rp.bs_fft_torch(t)) ** 2
        cosine = (pp * tt).sum(1) / (
            torch.linalg.vector_norm(pp, dim=1)
            * torch.linalg.vector_norm(tt, dim=1)
        ).clamp_min(1e-30)
        pas += float(cosine.sum())
        pas_n += cosine.numel()
        pp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
        tt = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
        cosine = (pp * tt).sum(-1) / (
            torch.linalg.vector_norm(pp, dim=-1)
            * torch.linalg.vector_norm(tt, dim=-1)
        ).clamp_min(1e-30)
        pdp += float(cosine.sum())
        pdp_n += cosine.numel()
        error += float(torch.sum(torch.abs(p - t) ** 2, dtype=torch.float64))
        energy += float(torch.sum(torch.abs(t) ** 2, dtype=torch.float64))
    c1, c2, c3 = pas / pas_n, pdp / pdp_n, error / energy
    return {
        "c1_pas": c1,
        "c2_pdp": c2,
        "c3_nmse": c3,
        "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1 + c3),
    }


def run() -> None:
    _, channel, _ = rp.load_data()
    folds = [fold_data(channel, fold) for fold in range(5)]
    groups = sorted(set(np.concatenate([row[1] for row in folds]).tolist()))
    requested_groups = {
        int(value) for value in os.environ.get("R2_CAL_GROUPS", "").split(",")
        if value.strip()
    }
    if requested_groups:
        groups = [group for group in groups if group in requested_groups]
    requested_names = {
        value for value in os.environ.get("R2_CAL_NAMES", "").split(",") if value
    }
    configs = [config for config in CONFIGS if not requested_names or config[0] in requested_names]
    sums = {}
    baselines = {}
    for fold, (_, labels, pred, target) in enumerate(folds):
        for group in groups:
            mask = labels == group
            sums[fold, group] = spectral_sums(pred[mask], target[mask])
            p = torch.as_tensor(np.asarray(pred[mask]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(target[mask]).copy(), device=DEVICE)
            baselines[fold, group] = torch_score(p, t)
            print(json.dumps({"stage": "sums", "fold": fold, "group": group}), flush=True)

    records = []
    for name, level, pas_blend, pdp_blend, order in configs:
        for fold, (_, labels, pred, target) in enumerate(folds):
            for group in groups:
                train_sums = [sums[other, group] for other in range(5) if other != fold]
                pas_ratio = ratio_profile(train_sums, "pas", level)
                pdp_ratio = ratio_profile(train_sums, "pdp", level)
                mask = labels == group
                p = torch.as_tensor(np.asarray(pred[mask]).copy(), device=DEVICE)
                t = torch.as_tensor(np.asarray(target[mask]).copy(), device=DEVICE)
                corrected = apply_correction(
                    p, pas_ratio, pdp_ratio, pas_blend, pdp_blend, order
                )
                score = torch_score(corrected, t)
                delta = score["score"] - baselines[fold, group]["score"]
                records.append(
                    {
                        "name": name,
                        "fold": fold,
                        "group": group,
                        "delta": delta,
                        **score,
                    }
                )
        print(json.dumps({"stage": "config", "name": name}), flush=True)

    summary = []
    for name, *_ in configs:
        for group in groups:
            selected = [
                row for row in records if row["name"] == name and row["group"] == group
            ]
            delta = np.asarray([row["delta"] for row in selected])
            summary.append(
                {
                    "name": name,
                    "group": group,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "positive": int(np.sum(delta > 0)),
                }
            )
    summary.sort(key=lambda row: row["mean_delta"], reverse=True)
    safe = [row for row in summary if row["min_delta"] > 0]
    payload = {"records": records, "summary": summary, "safe": safe}
    (ROOT / "matched_spectral_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:15], "safe": safe[:15]}), flush=True)


if __name__ == "__main__":
    run()
