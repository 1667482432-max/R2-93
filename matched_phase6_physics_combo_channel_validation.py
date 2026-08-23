from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp
ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BANDS = 24
BAND_WIDTH = 192 // BANDS
DESCRIPTOR_LABEL = os.environ.get("R2_PHYSICS_DESCRIPTOR_LABEL", "physics_combo")
CONFIGS = tuple(
    (scale, pdp_strength, iterations)
    for scale in (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
    for pdp_strength, iterations in ((1.5, 4), (1.5, 12))
)


def normalize_np(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def normalized(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=dim, keepdim=True).clamp_min(1e-30)


def update_scores(
    accumulator: np.ndarray,
    group_accumulator: np.ndarray | None,
    index: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    weights: torch.Tensor,
    labels: np.ndarray,
) -> None:
    pas = torch.abs(rp.bs_fft_torch(prediction)) ** 2
    pas_cos = (pas * target_pas).sum(1) / (
        torch.linalg.vector_norm(pas, dim=1)
        * torch.linalg.vector_norm(target_pas, dim=1)
    ).clamp_min(1e-30)
    pdp = torch.abs(torch.fft.fft(prediction, dim=-1, norm="ortho")) ** 2
    pdp_cos = (pdp * target_pdp).sum(-1) / (
        torch.linalg.vector_norm(pdp, dim=-1)
        * torch.linalg.vector_norm(target_pdp, dim=-1)
    ).clamp_min(1e-30)
    error = torch.abs(prediction - target) ** 2
    target_energy = torch.abs(target) ** 2

    accumulator[index, 0] += float((pas_cos * weights[:, None, None]).sum())
    accumulator[index, 1] += float((pdp_cos * weights[:, None, None]).sum())
    accumulator[index, 2] += float((error * weights[:, None, None, None]).sum(dtype=torch.float64))
    accumulator[index, 3] += float((target_energy * weights[:, None, None, None]).sum(dtype=torch.float64))
    accumulator[index, 4] += float(weights.sum()) * pas_cos.shape[1] * pas_cos.shape[2]
    accumulator[index, 5] += float(weights.sum()) * pdp_cos.shape[1] * pdp_cos.shape[2]

    if group_accumulator is not None:
        for group in np.unique(labels):
            local_np = np.flatnonzero(labels == group)
            local = torch.as_tensor(local_np, device=prediction.device)
            row = group_accumulator[index, int(group)]
            row[0] += float(pas_cos[local].sum())
            row[1] += float(pdp_cos[local].sum())
            row[2] += float(error[local].sum(dtype=torch.float64))
            row[3] += float(target_energy[local].sum(dtype=torch.float64))
            row[4] += len(local_np) * pas_cos.shape[1] * pas_cos.shape[2]
            row[5] += len(local_np) * pdp_cos.shape[1] * pdp_cos.shape[2]


def build_descriptors() -> None:
    gate = np.load(ROOT / "matched_phase6_pas_rich_gate_features.npz")
    for fold in range(5):
        path = ROOT / f"matched_phase6_{DESCRIPTOR_LABEL}_pas_band24_fold{fold}.npy"
        if path.exists():
            continue
        labels = gate[f"labels{fold}"]
        base = np.load(ROOT / f"matched_phase5_pas_band24_fold{fold}.npy", mmap_mode="r")
        horizontal = np.load(
            ROOT / f"matched_fitted_canonical_leaf8_mf8_pas_band24_fold{fold}.npy",
            mmap_mode="r",
        )
        vertical = np.load(
            ROOT / f"matched_fitted_vertical_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        safe = np.load(
            ROOT / f"matched_phase6_safe_combo_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        rich = np.load(
            ROOT / f"matched_rich_tree_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        desired = normalize_np(0.85 * base + 0.075 * horizontal + 0.075 * vertical)
        desired = normalize_np(desired + 1.5 * (safe - base))
        group6 = (labels == 6) & (gate[f"x{fold}"][:, 8] > 1.3770753145217889)
        desired[group6] = normalize_np(0.5 * desired[group6] + 0.5 * rich[group6])
        np.save(path, desired.astype(np.float32))
        print(
            json.dumps({"stage": "descriptor", "fold": fold, "group6_rows": int(group6.sum())}),
            flush=True,
        )


@torch.no_grad()
def run() -> None:
    build_descriptors()
    _, channel, _ = rp.load_data()
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    fold_records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        weights_np = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels], dtype=np.float64
        )
        base = np.load(ROOT / f"matched_phase5_full_fold{fold}.npy", mmap_mode="r")
        desired = np.load(
            ROOT / f"matched_phase6_{DESCRIPTOR_LABEL}_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        accumulator = np.zeros((1 + len(CONFIGS), 6), np.float64)
        group_accumulator = (
            np.zeros((1 + len(CONFIGS), 11, 6), np.float64)
            if os.environ.get("R2_GROUP_STATS") == "1"
            else None
        )
        for start in range(0, len(val), 4):
            stop = min(start + 4, len(val))
            p = torch.as_tensor(np.asarray(base[start:stop]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(channel[val[start:stop]]).copy(), device=DEVICE)
            w = torch.as_tensor(weights_np[start:stop], device=DEVICE)
            tt_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            tt_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
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
            base_pdp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
            update_scores(
                accumulator, group_accumulator, 0, p, t, tt_pas, tt_pdp, w, labels[start:stop]
            )
            for index, (scale, pdp_strength, iterations) in enumerate(CONFIGS, 1):
                target_band = normalized((1.0 - scale) * base_band + scale * desired_band, 1)
                epsilon = 1e-3 / base_band.shape[1]
                ratio = ((target_band + epsilon) / (base_band + epsilon)).clamp(0.25, 4.0)
                target_pas = base_pas * ratio.repeat_interleave(BAND_WIDTH, dim=3)
                x = rp.bs_ifft_torch(
                    base_pas_complex * torch.sqrt(target_pas / base_pas.clamp_min(1e-30))
                )
                for _ in range(iterations):
                    z = torch.fft.fft(x, dim=-1, norm="ortho")
                    correction = torch.sqrt(base_pdp).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
                    x = torch.fft.ifft(z * correction.pow(pdp_strength), dim=-1, norm="ortho")
                    z = rp.bs_fft_torch(x)
                    correction = torch.sqrt(target_pas).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
                    x = rp.bs_ifft_torch(z * correction)
                update_scores(
                    accumulator,
                    group_accumulator,
                    index,
                    x,
                    t,
                    tt_pas,
                    tt_pdp,
                    w,
                    labels[start:stop],
                )
        rows = []
        for index, config in enumerate(((0.0, 0.0, 0),) + CONFIGS):
            c1 = accumulator[index, 0] / accumulator[index, 4]
            c2 = accumulator[index, 1] / accumulator[index, 5]
            c3 = accumulator[index, 2] / accumulator[index, 3]
            rows.append(
                {
                    "scale": config[0],
                    "pdp_strength": config[1],
                    "iterations": config[2],
                    "c1_pas": c1,
                    "c2_pdp": c2,
                    "c3_nmse": c3,
                    "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3),
                }
            )
        group_rows = None
        if group_accumulator is not None:
            group_rows = {}
            for group in range(11):
                local_rows = []
                for index, config in enumerate(((0.0, 0.0, 0),) + CONFIGS):
                    values = group_accumulator[index, group]
                    c1 = values[0] / values[4]
                    c2 = values[1] / values[5]
                    c3 = values[2] / values[3]
                    local_rows.append(
                        {
                            "scale": config[0],
                            "pdp_strength": config[1],
                            "iterations": config[2],
                            "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3),
                        }
                    )
                group_rows[str(group)] = local_rows
        fold_records.append({"fold": fold, "rows": rows, "groups": group_rows})
        print(
            json.dumps({"stage": "fold", "fold": fold, "best": max(rows, key=lambda row: row["score"])}),
            flush=True,
        )

    baseline = np.asarray([row["rows"][0]["score"] for row in fold_records])
    summary = []
    for index, config in enumerate(CONFIGS, 1):
        values = np.asarray([row["rows"][index]["score"] for row in fold_records])
        delta = values - baseline
        summary.append(
            {
                "scale": config[0],
                "pdp_strength": config[1],
                "iterations": config[2],
                "scores": values.tolist(),
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    group_summary = {}
    if os.environ.get("R2_GROUP_STATS") == "1":
        for group in range(11):
            baseline_group = np.asarray(
                [row["groups"][str(group)][0]["score"] for row in fold_records]
            )
            rows = []
            for index, config in enumerate(CONFIGS, 1):
                values = np.asarray(
                    [row["groups"][str(group)][index]["score"] for row in fold_records]
                )
                delta = values - baseline_group
                rows.append(
                    {
                        "scale": config[0],
                        "pdp_strength": config[1],
                        "iterations": config[2],
                        "deltas": delta.tolist(),
                        "mean_delta": float(delta.mean()),
                        "min_delta": float(delta.min()),
                        "lcb": float(delta.mean() - 0.75 * delta.std()),
                    }
                )
            rows.sort(key=lambda row: row["lcb"], reverse=True)
            group_summary[str(group)] = rows
    (ROOT / f"matched_phase6_{DESCRIPTOR_LABEL}_channel_validation.json").write_text(
        json.dumps({"folds": fold_records, "summary": summary, "group_summary": group_summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:15]}), flush=True)


if __name__ == "__main__":
    run()
