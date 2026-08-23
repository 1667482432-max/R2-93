from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import r2_pipeline as rp
from matched_phase6_physics_combo_channel_validation import update_scores
from phase8_anchor_retained_pas_screen import horizontal_shifts, mapped_anchors
from phase8_anchor_retained_pdp_screen import official_geometry


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BANDS_VALUES = (24,)
ALIGNMENTS = ("none", "horizontal")
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
ITERATIONS = (12,)
CONFIGS = tuple(itertools.product(BANDS_VALUES, ALIGNMENTS, ALPHAS, ITERATIONS))


def normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


@torch.no_grad()
def pas_descriptor(channel: np.ndarray, bands: int, cache: Path) -> np.ndarray:
    if cache.exists():
        value = np.load(cache)
        if value.shape[-1] == bands:
            return value
    width = 192 // bands
    rows = []
    for start in range(0, len(channel), 4):
        stop = min(start + 4, len(channel))
        x = torch.as_tensor(np.asarray(channel[start:stop]).copy(), device=DEVICE)
        pas = torch.abs(rp.bs_fft_torch(x)) ** 2
        pas /= torch.linalg.vector_norm(pas, dim=1, keepdim=True).clamp_min(1e-30)
        if width > 1:
            pas = pas.reshape(len(x), 256, 4, bands, width).mean(-1)
            pas /= torch.linalg.vector_norm(pas, dim=1, keepdim=True).clamp_min(1e-30)
        rows.append(pas.cpu().numpy().astype(np.float32))
    value = np.concatenate(rows)
    np.save(cache, value)
    return value


def roll_horizontal(value: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    bands = value.shape[-1]
    shaped = value.reshape(len(value), 2, 16, 8, 4, bands)
    output = np.empty_like(shaped)
    for row, shift in enumerate(shifts):
        output[row] = np.roll(shaped[row], int(shift), axis=1)
    return output.reshape(value.shape)


def anchor_prediction(
    xy: np.ndarray,
    labels: np.ndarray,
    shifts: np.ndarray,
    anchors: np.ndarray,
    query: np.ndarray,
    truth: np.ndarray,
    alignment: str,
) -> np.ndarray:
    output = np.zeros((len(query), *truth.shape[1:]), dtype=np.float32)
    for group in np.unique(labels[query]):
        anchor = anchors[labels[anchors] == group]
        rows = np.flatnonzero(labels[query] == group)
        if not len(anchor):
            continue
        k = min(4, len(anchor))
        distance, local = cKDTree(xy[anchor]).query(xy[query[rows]], k=k)
        distance = np.asarray(distance)
        local = np.asarray(local)
        if distance.ndim == 1:
            distance = distance[:, None]
            local = local[:, None]
        scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
        weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** 2
        weight /= weight.sum(1, keepdims=True)
        for offset, row in enumerate(rows):
            chosen = anchor[local[offset]]
            value = truth[chosen]
            if alignment == "horizontal":
                value = roll_horizontal(value, shifts[query[row]] - shifts[chosen])
            output[row] = np.einsum(
                "k,kaub->aub", weight[offset], value, optimize=True
            )
    return output


def components(values: np.ndarray) -> dict[str, float]:
    c1 = values[0] / values[4]
    c2 = values[1] / values[5]
    c3 = values[2] / values[3]
    return {
        "c1_pas": float(c1),
        "c2_pdp": float(c2),
        "c3_nmse": float(c3),
        "score": float(0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3)),
    }


@torch.no_grad()
def project(
    channel: torch.Tensor,
    base_band: torch.Tensor,
    desired_band: torch.Tensor,
    bands: int,
    iterations: int,
) -> torch.Tensor:
    width = 192 // bands
    epsilon = 1e-3 / bands
    base_pas_complex = rp.bs_fft_torch(channel)
    base_pas = torch.abs(base_pas_complex) ** 2
    ratio = ((desired_band + epsilon) / (base_band + epsilon)).clamp(0.25, 4.0)
    if width > 1:
        ratio = ratio.repeat_interleave(width, dim=3)
    target_pas = base_pas * ratio
    value = rp.bs_ifft_torch(
        base_pas_complex * torch.sqrt(target_pas / base_pas.clamp_min(1e-30))
    )
    base_pdp = torch.abs(torch.fft.fft(channel, dim=-1, norm="ortho")) ** 2
    for _ in range(iterations):
        z = torch.fft.fft(value, dim=-1, norm="ortho")
        correction = torch.sqrt(base_pdp).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
        value = torch.fft.ifft(z * correction.pow(1.5), dim=-1, norm="ortho")
        z = rp.bs_fft_torch(value)
        correction = torch.sqrt(target_pas).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
        value = rp.bs_ifft_torch(z * correction)
    return value


def run() -> None:
    pos, channel, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    official_counts, actual_fraction, actual_counts = official_geometry(pos, energy, test_pos)
    all_shifts = horizontal_shifts(pos)
    fold_rows = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        query = np.setdiff1d(np.arange(len(val)), anchors)
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        descriptor_map = {}
        for bands in BANDS_VALUES:
            base = pas_descriptor(
                prediction,
                bands,
                ROOT / f"phase8_anchor_retained_fold{fold}_base_pas_band{bands}.npy",
            )
            truth = pas_descriptor(
                channel[val],
                bands,
                ROOT / f"phase8_anchor_retained_fold{fold}_truth_pas_band{bands}.npy",
            )
            for alignment in ALIGNMENTS:
                anchor = anchor_prediction(
                    pos[val, :2], labels, all_shifts[val], anchors, query, truth, alignment
                )
                descriptor_map[bands, alignment] = (base, anchor)

        accumulator = np.zeros((1 + len(CONFIGS), 6), np.float64)
        group_accumulator = np.zeros((1 + len(CONFIGS), 11, 6), np.float64)
        weights_np = np.asarray(
            [official_counts[int(group)] / np.sum(labels[query] == group) for group in labels[query]],
            dtype=np.float32,
        )
        for start in range(0, len(query), 2):
            stop = min(start + 2, len(query))
            local = query[start:stop]
            p = torch.as_tensor(np.asarray(prediction[local]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(channel[val[local]]).copy(), device=DEVICE)
            weights = torch.as_tensor(weights_np[start:stop], device=DEVICE)
            truth_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            truth_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
            update_scores(
                accumulator, group_accumulator, 0, p, t, truth_pas, truth_pdp,
                weights, labels[local]
            )
            for config_index, (bands, alignment, alpha, iterations) in enumerate(CONFIGS, 1):
                base, anchor = descriptor_map[bands, alignment]
                base_band = torch.as_tensor(base[local].copy(), device=DEVICE)
                desired_np = normalize(
                    (1.0 - alpha) * base[local] + alpha * anchor[start:stop]
                ).astype(np.float32)
                desired = torch.as_tensor(desired_np, device=DEVICE)
                value = project(p, base_band, desired, bands, iterations)
                update_scores(
                    accumulator, group_accumulator, config_index, value, t,
                    truth_pas, truth_pdp, weights, labels[local]
                )
        baseline = components(accumulator[0])
        represented_groups = sorted(set(int(group) for group in labels[query]))
        baseline_groups = {
            str(group): components(group_accumulator[0, group])
            for group in represented_groups
        }
        rows = []
        for config_index, config in enumerate(CONFIGS, 1):
            value = components(accumulator[config_index])
            group_values = {
                str(group): components(group_accumulator[config_index, group])
                for group in represented_groups
            }
            rows.append(
                {
                    "bands": config[0], "alignment": config[1], "alpha": config[2],
                    "iterations": config[3], **value,
                    "delta": value["score"] - baseline["score"],
                    "group_components": group_values,
                    "group_deltas": {
                        str(group): group_values[str(group)]["score"]
                        - baseline_groups[str(group)]["score"]
                        for group in represented_groups
                    },
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "baseline": baseline,
                "baseline_groups": baseline_groups,
                "query_group_counts": {
                    str(group): int(np.sum(labels[query] == group))
                    for group in represented_groups
                },
                "rows": rows,
            }
        )
        print(json.dumps({"stage": "fold", "fold": fold, "best": max(rows, key=lambda x: x["delta"])}), flush=True)

    summary = []
    for index, config in enumerate(CONFIGS):
        scores = [fold["rows"][index]["score"] for fold in fold_rows]
        deltas = [fold["rows"][index]["delta"] for fold in fold_rows]
        summary.append(
            {
                "bands": config[0], "alignment": config[1], "alpha": config[2],
                "iterations": config[3], "scores": scores, "deltas": deltas,
                "mean_delta": float(np.mean(deltas)), "min_delta": float(np.min(deltas)),
                "lcb": float(np.mean(deltas) - np.std(deltas)),
            }
        )
    summary.sort(key=lambda row: (row["lcb"], row["mean_delta"]), reverse=True)
    output = {
        "official_test_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "folds": fold_rows,
        "summary": summary,
    }
    (ROOT / "phase8_anchor_retained_pas_resolution_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "top": summary[:20]}), flush=True)


if __name__ == "__main__":
    run()
