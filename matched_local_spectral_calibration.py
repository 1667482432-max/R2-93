from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import r2_pipeline as rp
from matched_spectral_calibration import apply_correction, torch_score


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
CONFIGS = tuple(
    (f"local_k{k}_b{int(blend * 100):02d}", k, blend)
    for k in (5, 10, 20)
    for blend in (0.05, 0.10, 0.20)
)


@torch.no_grad()
def residual_profiles(pred, target):
    pas_rows = []
    pdp_rows = []
    for start in range(0, len(pred), 4):
        stop = min(start + 4, len(pred))
        p = torch.as_tensor(np.asarray(pred[start:stop]).copy(), device=DEVICE)
        t = torch.as_tensor(np.asarray(target[start:stop]).copy(), device=DEVICE)
        pp = torch.abs(rp.bs_fft_torch(p)) ** 2
        tt = torch.abs(rp.bs_fft_torch(t)) ** 2
        pp /= pp.sum(1, keepdim=True).clamp_min(1e-30)
        tt /= tt.sum(1, keepdim=True).clamp_min(1e-30)
        pp = pp.mean(3)
        tt = tt.mean(3)
        epsilon = 1e-3 / pp.shape[1]
        log_ratio = torch.log(tt + epsilon) - torch.log(pp + epsilon)
        log_ratio -= log_ratio.mean(1, keepdim=True)
        pas_rows.append(log_ratio.clamp(-1.3863, 1.3863).cpu().numpy())

        pp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
        tt = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
        pp /= pp.sum(-1, keepdim=True).clamp_min(1e-30)
        tt /= tt.sum(-1, keepdim=True).clamp_min(1e-30)
        pp = pp.mean(1)
        tt = tt.mean(1)
        epsilon = 1e-3 / pp.shape[-1]
        log_ratio = torch.log(tt + epsilon) - torch.log(pp + epsilon)
        log_ratio -= log_ratio.mean(-1, keepdim=True)
        pdp_rows.append(log_ratio.clamp(-1.3863, 1.3863).cpu().numpy())
    return np.concatenate(pas_rows), np.concatenate(pdp_rows)


def predict_profiles(train_xy, train_pas, train_pdp, query_xy, neighbors):
    k = min(neighbors, len(train_xy))
    distance, index = cKDTree(train_xy).query(query_xy, k=k)
    distance = np.asarray(distance)
    index = np.asarray(index)
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]
    scale = max(float(np.median(distance[:, -1])), 1.0)
    weight = 1.0 / (distance + 0.25 * scale) ** 2
    weight /= weight.sum(1, keepdims=True)
    pas = np.sum(train_pas[index] * weight[:, :, None, None], axis=1)
    pdp = np.sum(train_pdp[index] * weight[:, :, None, None], axis=1)
    return np.exp(pas).astype(np.float32), np.exp(pdp).astype(np.float32)


@torch.no_grad()
def apply_per_sample(x, pas_ratio, pdp_ratio, blend):
    pas = torch.as_tensor(pas_ratio, device=DEVICE)
    pdp = torch.as_tensor(pdp_ratio, device=DEVICE)
    z = rp.bs_fft_torch(x)
    z *= pas[:, :, :, None].pow(0.5 * blend)
    x = rp.bs_ifft_torch(z)
    z = torch.fft.fft(x, dim=-1, norm="ortho")
    z *= pdp[:, None, :, :].pow(0.5 * blend)
    return torch.fft.ifft(z, dim=-1, norm="ortho")


def run() -> None:
    pos, channel, _ = rp.load_data()
    folds = []
    groups = set()
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        pred = np.load(ROOT / f"matched_phase2_full_fold{fold}.npy", mmap_mode="r")
        target = channel[val]
        pas, pdp = residual_profiles(pred, target)
        folds.append(
            {
                "fold": fold,
                "val": val,
                "labels": labels,
                "pred": pred,
                "target": target,
                "xy": pos[val, :2],
                "pas": pas,
                "pdp": pdp,
            }
        )
        groups.update(labels.tolist())
        print(json.dumps({"stage": "profile", "fold": fold}), flush=True)
    groups = sorted(groups)

    baselines = {}
    for row in folds:
        for group in groups:
            mask = row["labels"] == group
            p = torch.as_tensor(np.asarray(row["pred"][mask]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(row["target"][mask]).copy(), device=DEVICE)
            baselines[row["fold"], group] = torch_score(p, t)

    records = []
    for name, neighbors, blend in CONFIGS:
        for row in folds:
            for group in groups:
                train = [other for other in folds if other["fold"] != row["fold"]]
                train_xy = np.concatenate(
                    [other["xy"][other["labels"] == group] for other in train]
                )
                train_pas = np.concatenate(
                    [other["pas"][other["labels"] == group] for other in train]
                )
                train_pdp = np.concatenate(
                    [other["pdp"][other["labels"] == group] for other in train]
                )
                mask = row["labels"] == group
                pas_ratio, pdp_ratio = predict_profiles(
                    train_xy, train_pas, train_pdp, row["xy"][mask], neighbors
                )
                p = torch.as_tensor(np.asarray(row["pred"][mask]).copy(), device=DEVICE)
                t = torch.as_tensor(np.asarray(row["target"][mask]).copy(), device=DEVICE)
                corrected = apply_per_sample(p, pas_ratio, pdp_ratio, blend)
                score = torch_score(corrected, t)
                records.append(
                    {
                        "name": name,
                        "fold": row["fold"],
                        "group": group,
                        "delta": score["score"] - baselines[row["fold"], group]["score"],
                        **score,
                    }
                )
        print(json.dumps({"stage": "config", "name": name}), flush=True)

    summary = []
    for name, *_ in CONFIGS:
        for group in groups:
            delta = np.asarray(
                [
                    row["delta"]
                    for row in records
                    if row["name"] == name and row["group"] == group
                ]
            )
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
    (ROOT / "matched_local_spectral_calibration.json").write_text(
        json.dumps({"records": records, "summary": summary, "safe": safe}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"top": summary[:15], "safe": safe[:15]}), flush=True)


if __name__ == "__main__":
    run()
