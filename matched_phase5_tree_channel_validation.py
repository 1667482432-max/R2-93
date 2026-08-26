from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
FINE = os.environ.get("R2_TREE_FINE", "0") == "1"
MODE = os.environ.get("R2_TREE_PAS_MODE", "global")
ALPHAS = (
    (0.50, 0.75, 1.00, 1.25)
    if MODE == "ensemble"
    else ((0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50) if FINE else (0.05, 0.10, 0.20, 0.35))
)
PDP_RESTORE = (0.50, 0.75, 1.00, 1.25, 1.50) if FINE else (0.0, 0.25, 0.50, 1.0)
FINAL_PAS = (0.25, 0.50, 0.75, 1.00) if FINE else (0.0, 0.25, 0.50)
CONFIGS = tuple(
    (alpha, pdp_restore, final_pas)
    for alpha in ALPHAS
    for pdp_restore in PDP_RESTORE
    for final_pas in FINAL_PAS
)
BANDS = int(os.environ.get("R2_BANDS", "12"))
BAND_WIDTH = 192 // BANDS


def normalized(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=dim, keepdim=True).clamp_min(1e-30)


def update_score(
    accumulator: np.ndarray,
    index: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_pas: torch.Tensor,
    target_pdp: torch.Tensor,
    weights: torch.Tensor,
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
    accumulator[index, 0] += float((pas_cos * weights[:, None, None]).sum())
    accumulator[index, 1] += float((pdp_cos * weights[:, None, None]).sum())
    accumulator[index, 2] += float(
        (torch.abs(prediction - target) ** 2 * weights[:, None, None, None]).sum(
            dtype=torch.float64
        )
    )
    accumulator[index, 3] += float(
        (torch.abs(target) ** 2 * weights[:, None, None, None]).sum(dtype=torch.float64)
    )
    accumulator[index, 4] += float(weights.sum()) * pas_cos.shape[1] * pas_cos.shape[2]
    accumulator[index, 5] += float(weights.sum()) * pdp_cos.shape[1] * pdp_cos.shape[2]


@torch.no_grad()
def run() -> None:
    _, channel, _ = rp.load_data()
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    fold_records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        weights_np = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels],
            dtype=np.float64,
        )
        base = np.load(ROOT / f"matched_phase4_full_fold{fold}.npy", mmap_mode="r")
        tree_name = (
            f"matched_tree_pas_band{BANDS}_fold{fold}.npy"
            if MODE in ("band12", "ensemble")
            else (
                f"matched_tree_pas_ue_fold{fold}.npy"
                if MODE == "ue"
                else f"matched_tree_pas_fold{fold}.npy"
            )
        )
        tree = np.load(ROOT / tree_name, mmap_mode="r")
        rbf = (
            np.load(ROOT / f"matched_rbf_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
            if MODE == "ensemble"
            else None
        )
        accumulator = np.zeros((1 + len(CONFIGS), 6), np.float64)
        for start in range(0, len(val), 4):
            stop = min(start + 4, len(val))
            p = torch.as_tensor(np.asarray(base[start:stop]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(channel[val[start:stop]]).copy(), device=DEVICE)
            w = torch.as_tensor(weights_np[start:stop], device=DEVICE)
            tt_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            tt_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
            base_pas_complex = rp.bs_fft_torch(p)
            base_pas = torch.abs(base_pas_complex) ** 2
            if MODE in ("band12", "ensemble"):
                base_aggregate = normalized(
                    normalized(base_pas, 1)
                    .reshape(len(p), 256, 4, BANDS, BAND_WIDTH)
                    .mean(4),
                    1,
                )
            elif MODE == "ue":
                base_aggregate = normalized(normalized(base_pas, 1).mean(3), 1)
            else:
                base_aggregate = normalized(base_pas.sum((2, 3)), 1)
            tree_pas = torch.as_tensor(np.asarray(tree[start:stop]).copy(), device=DEVICE)
            if MODE == "ensemble":
                rbf_pas = torch.as_tensor(np.asarray(rbf[start:stop]).copy(), device=DEVICE)
                tree_pas = normalized(
                    0.60 * base_aggregate + 0.25 * tree_pas + 0.15 * rbf_pas,
                    1,
                )
            base_pdp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
            update_score(accumulator, 0, p, t, tt_pas, tt_pdp, w)

            for config_index, (alpha, pdp_restore, final_pas) in enumerate(CONFIGS, 1):
                desired = normalized((1.0 - alpha) * base_aggregate + alpha * tree_pas, 1)
                epsilon = 1e-3 / base_aggregate.shape[1]
                ratio = ((desired + epsilon) / (base_aggregate + epsilon)).clamp(0.25, 4.0)
                if MODE in ("band12", "ensemble"):
                    ratio_full = ratio.repeat_interleave(BAND_WIDTH, dim=3)
                elif MODE == "ue":
                    ratio_full = ratio[:, :, :, None]
                else:
                    ratio_full = ratio[:, :, None, None]
                target_pas = base_pas * ratio_full
                x = rp.bs_ifft_torch(base_pas_complex * torch.sqrt(ratio_full))
                if pdp_restore:
                    z = torch.fft.fft(x, dim=-1, norm="ortho")
                    correction = torch.sqrt(base_pdp).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
                    x = torch.fft.ifft(z * correction.pow(pdp_restore), dim=-1, norm="ortho")
                if final_pas:
                    z = rp.bs_fft_torch(x)
                    correction = torch.sqrt(target_pas).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
                    x = rp.bs_ifft_torch(z * correction.pow(final_pas))
                update_score(accumulator, config_index, x, t, tt_pas, tt_pdp, w)
        rows = []
        for index, config in enumerate(((0.0, 0.0, 0.0),) + CONFIGS):
            c1 = accumulator[index, 0] / accumulator[index, 4]
            c2 = accumulator[index, 1] / accumulator[index, 5]
            c3 = accumulator[index, 2] / accumulator[index, 3]
            rows.append(
                {
                    "alpha": config[0],
                    "pdp_restore": config[1],
                    "final_pas": config[2],
                    "c1_pas": c1,
                    "c2_pdp": c2,
                    "c3_nmse": c3,
                    "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3),
                }
            )
        fold_records.append({"fold": fold, "rows": rows})
        best = max(rows, key=lambda row: row["score"])
        print(json.dumps({"stage": "fold", "fold": fold, "best": best}), flush=True)

    summary = []
    baseline = np.asarray([record["rows"][0]["score"] for record in fold_records])
    for config_index, config in enumerate(CONFIGS, 1):
        values = np.asarray([record["rows"][config_index]["score"] for record in fold_records])
        delta = values - baseline
        summary.append(
            {
                "alpha": config[0],
                "pdp_restore": config[1],
                "final_pas": config[2],
                "scores": values.tolist(),
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    payload = {"folds": fold_records, "summary": summary}
    output_name = (
        f"matched_phase5_tree_band{BANDS}_channel_validation.json"
        if MODE == "band12"
        else (
            f"matched_phase5_tree_rbf_band{BANDS}_channel_validation.json"
            if MODE == "ensemble"
            else (
            "matched_phase5_tree_ue_channel_validation.json"
            if MODE == "ue"
            else "matched_phase5_tree_channel_validation.json"
            )
        )
    )
    if FINE:
        output_name = output_name.replace(".json", "_fine.json")
    (ROOT / output_name).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:20]}), flush=True)


if __name__ == "__main__":
    run()
