from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

import r2_pipeline as rp
from matched_local_spectral_calibration import apply_per_sample, residual_profiles
from matched_phase5_feature_residual import feature_matrix, predict_ratios, prediction_summaries
from matched_phase5_tree_channel_validation import update_score
from matched_phase5_tree_pdp_ue_descriptor import (
    base_descriptor as pdp_base_descriptor,
    normalize_last as normalize_pdp,
)


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BANDS = 24
BAND_WIDTH = 192 // BANDS
SAFE = {
    0: ("geo_spec", 10, 0.10, "both"),
    1: ("geo_spec", 5, 0.10, "pdp"),
    5: ("xy", 10, 0.20, "pdp"),
}
SUBSETS = tuple(
    subset
    for count in range(4)
    for subset in itertools.combinations(sorted(SAFE), count)
)
PROJECTIONS = ((0.90, 1.50, 24), (1.00, 1.50, 24), (1.10, 1.50, 24))
PDP_TREE_ALPHAS = (0.00,)
CONFIGS = tuple(
    (projection, subset, pdp_tree_alpha)
    for projection in PROJECTIONS
    for subset in SUBSETS
    for pdp_tree_alpha in PDP_TREE_ALPHAS
)


def normalized(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=dim, keepdim=True).clamp_min(1e-30)


def prepare_folds() -> list[dict]:
    pos, channel, _ = rp.load_data()
    map_features = np.load(ROOT / "los_map_features.npy")[: len(pos)]
    folds = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        pred = np.load(ROOT / f"matched_phase4_full_fold{fold}.npy", mmap_mode="r")
        pas, pdp = residual_profiles(pred, channel[val])
        folds.append(
            {
                "fold": fold,
                "val": val,
                "labels": np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64),
                "pred": pred,
                "target": channel[val],
                "xy": pos[val, :2],
                "map": map_features[val],
                "spec": prediction_summaries(pred),
                "pas": pas,
                "pdp": pdp,
            }
        )
        print(json.dumps({"stage": "features", "fold": fold}), flush=True)
    return folds


def correction_ratios(folds: list[dict]) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    ratios = {}
    for row in folds:
        train_rows = [other for other in folds if other["fold"] != row["fold"]]
        for group, (feature_name, neighbors, _, axis) in SAFE.items():
            train_masks = [other["labels"] == group for other in train_rows]
            query_mask = row["labels"] == group
            train_feature, query_feature = feature_matrix(
                feature_name, train_rows, row, train_masks, query_mask
            )
            train_pas = np.concatenate(
                [other["pas"][mask] for other, mask in zip(train_rows, train_masks)]
            )
            train_pdp = np.concatenate(
                [other["pdp"][mask] for other, mask in zip(train_rows, train_masks)]
            )
            pas, pdp = predict_ratios(
                train_feature, query_feature, train_pas, train_pdp, neighbors
            )
            if axis == "pdp":
                pas = np.ones_like(pas)
            ratios[row["fold"], group] = pas, pdp
    return ratios


@torch.no_grad()
def run() -> None:
    folds = prepare_folds()
    ratios = correction_ratios(folds)
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    fold_records = []
    for row in folds:
        fold = row["fold"]
        labels = row["labels"]
        weights_np = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels], dtype=np.float64
        )
        tree = np.load(ROOT / f"matched_tree_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
        rbf = np.load(ROOT / f"matched_rbf_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
        pdp_base = pdp_base_descriptor(row["pred"])
        pdp_tree = np.load(ROOT / f"matched_tree_pdp_ue_fold{fold}.npy", mmap_mode="r")
        pdp_tree_ratios = {}
        for alpha in PDP_TREE_ALPHAS[1:]:
            desired = normalize_pdp((1.0 - alpha) * pdp_base + alpha * pdp_tree)
            pdp_tree_ratios[alpha] = np.clip(
                (desired + 1e-6) / (pdp_base + 1e-6), 0.5, 2.0
            ).astype(np.float32)
        accumulator = np.zeros((1 + len(CONFIGS), 6), np.float64)
        selected_output = (
            np.lib.format.open_memmap(
                ROOT / f"matched_phase5_full_fold{fold}.npy",
                mode="w+",
                dtype=np.complex64,
                shape=row["pred"].shape,
            )
            if os.environ.get("R2_SAVE_PHASE5_FOLDS")
            else None
        )
        group_offsets = {
            group: np.cumsum(np.r_[0, labels == group])[:-1] for group in SAFE
        }
        for start in range(0, len(labels), 4):
            stop = min(start + 4, len(labels))
            p = torch.as_tensor(np.asarray(row["pred"][start:stop]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(row["target"][start:stop]).copy(), device=DEVICE)
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
            tree_pas = torch.as_tensor(np.asarray(tree[start:stop]).copy(), device=DEVICE)
            rbf_pas = torch.as_tensor(np.asarray(rbf[start:stop]).copy(), device=DEVICE)
            ensemble = normalized(0.60 * base_band + 0.25 * tree_pas + 0.15 * rbf_pas, 1)
            base_pdp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
            update_score(accumulator, 0, p, t, tt_pas, tt_pdp, w)
            projected = {}
            for projection in PROJECTIONS:
                scale, pdp_strength, iterations = projection
                desired = normalized((1.0 - scale) * base_band + scale * ensemble, 1)
                epsilon = 1e-3 / base_band.shape[1]
                ratio = ((desired + epsilon) / (base_band + epsilon)).clamp(0.25, 4.0)
                target_pas = base_pas * ratio.repeat_interleave(BAND_WIDTH, dim=3)
                x = rp.bs_ifft_torch(base_pas_complex * torch.sqrt(target_pas / base_pas.clamp_min(1e-30)))
                for _ in range(iterations):
                    z = torch.fft.fft(x, dim=-1, norm="ortho")
                    correction = torch.sqrt(base_pdp).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
                    x = torch.fft.ifft(z * correction.pow(pdp_strength), dim=-1, norm="ortho")
                    z = rp.bs_fft_torch(x)
                    correction = torch.sqrt(target_pas).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
                    x = rp.bs_ifft_torch(z * correction)
                projected[projection] = x
            for config_index, (projection, subset, pdp_tree_alpha) in enumerate(CONFIGS, 1):
                x = projected[projection].clone()
                for group in subset:
                    local = np.flatnonzero(labels[start:stop] == group)
                    if not len(local):
                        continue
                    indices = group_offsets[group][start + local]
                    pas_ratio, pdp_ratio = ratios[fold, group]
                    blend = SAFE[group][2]
                    corrected = apply_per_sample(
                        x[torch.as_tensor(local, device=DEVICE)],
                        pas_ratio[indices],
                        pdp_ratio[indices],
                        blend,
                    )
                    x[torch.as_tensor(local, device=DEVICE)] = corrected
                if pdp_tree_alpha:
                    local = np.flatnonzero(labels[start:stop] == 5)
                    if len(local):
                        ratio = torch.as_tensor(
                            pdp_tree_ratios[pdp_tree_alpha][start + local], device=DEVICE
                        )
                        local_tensor = torch.as_tensor(local, device=DEVICE)
                        z = torch.fft.fft(x[local_tensor], dim=-1, norm="ortho")
                        z *= torch.sqrt(ratio)[:, None, :, :]
                        x[local_tensor] = torch.fft.ifft(z, dim=-1, norm="ortho")
                if (
                    selected_output is not None
                    and projection == (1.0, 1.5, 24)
                    and subset == (0, 1, 5)
                    and pdp_tree_alpha == 0.0
                ):
                    selected_output[start:stop] = x.cpu().numpy().astype(np.complex64)
                update_score(accumulator, config_index, x, t, tt_pas, tt_pdp, w)
        if selected_output is not None:
            selected_output.flush()
            del selected_output
        rows = []
        for index, config in enumerate((((0.0, 0.0, 0), (), 0.0),) + CONFIGS):
            c1 = accumulator[index, 0] / accumulator[index, 4]
            c2 = accumulator[index, 1] / accumulator[index, 5]
            c3 = accumulator[index, 2] / accumulator[index, 3]
            rows.append(
                {
                    "projection": config[0],
                    "groups": config[1],
                    "pdp_tree_alpha": config[2],
                    "c1_pas": c1,
                    "c2_pdp": c2,
                    "c3_nmse": c3,
                    "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3),
                }
            )
        fold_records.append({"fold": fold, "rows": rows})
        print(json.dumps({"stage": "fold", "fold": fold, "best": max(rows, key=lambda item: item["score"])}), flush=True)

    baseline = np.asarray([record["rows"][0]["score"] for record in fold_records])
    summary = []
    for config_index, config in enumerate(CONFIGS, 1):
        values = np.asarray([record["rows"][config_index]["score"] for record in fold_records])
        delta = values - baseline
        summary.append(
            {
                "projection": config[0],
                "groups": config[1],
                "pdp_tree_alpha": config[2],
                "scores": values.tolist(),
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
    summary.sort(key=lambda item: item["lcb"], reverse=True)
    (ROOT / "matched_phase5_composite_validation.json").write_text(
        json.dumps({"folds": fold_records, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:20]}), flush=True)


if __name__ == "__main__":
    run()
