from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
from matched_phase6_physics_combo_channel_validation import update_scores
from phase8_anchor_local_gate_channel_validation import ALPHA_GRID, components, prepare_folds
from phase8_anchor_retained_pas_screen import mapped_anchors, normalize
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_anchor_residual_pas_screen import residual_family


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
PAS_DESCRIPTORS = (
    ("safe", "frequency_mean", 0.20, 0.50, "residual_then_local"),
    ("threshold", "frequency_mean", 0.25, 0.75, "residual_then_local"),
)
PDP_ALPHAS = (0.0, 0.025, 0.05, 0.075, 0.10)
ITERATIONS = (4, 12)
CONFIGS = tuple(itertools.product(PAS_DESCRIPTORS, PDP_ALPHAS, ITERATIONS))


def normalize_last(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


def pdp_log_prediction(
    pos: np.ndarray,
    val: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    query_local: np.ndarray,
    base_all: np.ndarray,
    truth_all: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(np.asarray(base_all[query_local]), dtype=np.float32)
    epsilon = 1e-4 / 192
    for group in np.unique(labels[query_local]):
        rows = np.flatnonzero(labels[query_local] == group)
        group_anchors = anchors[labels[anchors] == group]
        if len(group_anchors) == 0:
            continue
        k = min(4, len(group_anchors))
        distance, local = cKDTree(pos[val[group_anchors], :2]).query(
            pos[val[query_local[rows]], :2], k=k
        )
        distance = np.asarray(distance)
        local = np.asarray(local)
        if k == 1:
            distance = distance[:, None]
            local = local[:, None]
        scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
        weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** 0.5
        weight /= weight.sum(1, keepdims=True)
        anchor_truth = np.asarray(truth_all[group_anchors])
        anchor_base = np.asarray(base_all[group_anchors])
        anchor_log = np.clip(
            np.log((anchor_truth + epsilon) / (anchor_base + epsilon)), -2.0, 2.0
        )
        output[rows] = np.einsum(
            "rk,rkaus->raus", weight, anchor_log[local], optimize=True
        )
    return output


@torch.no_grad()
def project_joint(
    channel: torch.Tensor,
    base_pas_band: torch.Tensor,
    desired_pas_band: torch.Tensor,
    desired_pdp: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    base_pas_complex = rp.bs_fft_torch(channel)
    base_pas = torch.abs(base_pas_complex) ** 2
    pas_ratio = ((desired_pas_band + 1e-3 / 24) / (base_pas_band + 1e-3 / 24)).clamp(0.25, 4.0)
    target_pas = base_pas * pas_ratio.repeat_interleave(8, dim=3)
    value = rp.bs_ifft_torch(
        base_pas_complex * torch.sqrt(target_pas / base_pas.clamp_min(1e-30))
    )
    base_pdp_complex = torch.fft.fft(channel, dim=-1, norm="ortho")
    base_pdp = torch.abs(base_pdp_complex) ** 2
    base_pdp_norm = base_pdp / torch.linalg.vector_norm(
        base_pdp, dim=-1, keepdim=True
    ).clamp_min(1e-30)
    pdp_ratio = ((desired_pdp + 1e-4 / 192) / (base_pdp_norm + 1e-4 / 192)).clamp(0.25, 4.0)
    target_pdp = base_pdp * pdp_ratio
    for _ in range(iterations):
        z = torch.fft.fft(value, dim=-1, norm="ortho")
        correction = torch.sqrt(target_pdp).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
        value = torch.fft.ifft(z * correction.pow(1.5), dim=-1, norm="ortho")
        z = rp.bs_fft_torch(value)
        correction = torch.sqrt(target_pas).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
        value = rp.bs_ifft_torch(z * correction)
    return value


@torch.no_grad()
def run() -> None:
    folds, pos, channel, energy, official_counts, actual_counts = prepare_folds()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    for fold, row in enumerate(folds):
        anchors = mapped_anchors(pos, row["val"], row["labels"], actual_fraction, official_counts)
        family = residual_family(
            pos, row["val"], row["labels"], anchors, row["query_local"],
            row["base_all"], target,
        )
        raw = family[16, 1.0, "log_ratio"]
        row["pas_residual"] = np.repeat(raw.mean(3, keepdims=True), raw.shape[3], axis=3)
        base_pdp_all = np.load(
            ROOT / f"phase9_anchor_fold{fold}_base_pdp_band192.npy", mmap_mode="r"
        )
        truth_pdp_all = np.load(
            ROOT / f"phase9_anchor_fold{fold}_truth_pdp_band192.npy", mmap_mode="r"
        )
        row["base_pdp"] = np.asarray(base_pdp_all[row["query_local"]])
        row["pdp_log"] = pdp_log_prediction(
            pos, row["val"], row["labels"], anchors, row["query_local"],
            base_pdp_all, truth_pdp_all,
        )

    gate_alpha = []
    for heldout in range(5):
        train_x = np.concatenate([folds[i]["x"] for i in range(5) if i != heldout])
        train_y = np.concatenate([folds[i]["gain_grid"] for i in range(5) if i != heldout])
        train_w = np.concatenate([folds[i]["weights"] for i in range(5) if i != heldout])
        model = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=80, max_features=0.7,
            n_jobs=-1, random_state=52180,
        )
        model.fit(train_x, train_y, sample_weight=train_w)
        gate_alpha.append(ALPHA_GRID[np.argmax(model.predict(folds[heldout]["x"]), axis=1)])

    fold_rows = []
    for fold, row in enumerate(folds):
        desired_pas = {}
        for descriptor in PAS_DESCRIPTORS:
            _, _, residual_alpha, local_scale, order = descriptor
            alpha = np.clip(local_scale * gate_alpha[fold], 0.0, 0.6)[:, None, None, None]
            correction = np.exp(residual_alpha * row["pas_residual"])
            if order == "residual_then_local":
                corrected = normalize(row["base"] * correction)
                value = normalize((1.0 - alpha) * corrected + alpha * row["local"])
            else:
                value = normalize(
                    ((1.0 - alpha) * row["base"] + alpha * row["local"]) * correction
                )
            desired_pas[descriptor] = value.astype(np.float32)
        desired_pdp = {
            alpha: normalize_last(row["base_pdp"] * np.exp(alpha * row["pdp_log"])).astype(np.float32)
            for alpha in PDP_ALPHAS
        }
        labels = row["labels"]
        query_local = row["query_local"]
        query = row["query"]
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        accumulator = np.zeros((1 + len(CONFIGS), 6), np.float64)
        group_accumulator = np.zeros((1 + len(CONFIGS), 11, 6), np.float64)
        for start in range(0, len(query), 2):
            stop = min(start + 2, len(query))
            local_query = query_local[start:stop]
            p = torch.as_tensor(np.asarray(prediction[local_query]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(channel[query[start:stop]]).copy(), device=DEVICE)
            weights = torch.as_tensor(row["weights"][start:stop].astype(np.float32), device=DEVICE)
            truth_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            truth_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
            update_scores(
                accumulator, group_accumulator, 0, p, t, truth_pas, truth_pdp,
                weights, labels[local_query]
            )
            base_pas_band = torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE)
            for index, (pas_descriptor, pdp_alpha, iterations) in enumerate(CONFIGS, 1):
                value = project_joint(
                    p,
                    base_pas_band,
                    torch.as_tensor(desired_pas[pas_descriptor][start:stop], device=DEVICE),
                    torch.as_tensor(desired_pdp[pdp_alpha][start:stop], device=DEVICE),
                    iterations,
                )
                update_scores(
                    accumulator, group_accumulator, index, value, t, truth_pas,
                    truth_pdp, weights, labels[local_query]
                )
        baseline = components(accumulator[0])
        rows = []
        for index, config in enumerate(CONFIGS, 1):
            value = components(accumulator[index])
            rows.append(
                {
                    "pas_label": config[0][0], "pas_descriptor": config[0],
                    "pdp_alpha": config[1], "iterations": config[2], **value,
                    "delta": value["score"] - baseline["score"],
                    "component_deltas": {
                        key: value[key] - baseline[key]
                        for key in ("c1_pas", "c2_pdp", "c3_nmse")
                    },
                }
            )
        fold_rows.append({"fold": fold, "baseline": baseline, "rows": rows})
        print(json.dumps({"stage": "fold", "fold": fold, "best": max(rows, key=lambda x: x["delta"])}), flush=True)

    summary = []
    for index, config in enumerate(CONFIGS):
        deltas = [fold["rows"][index]["delta"] for fold in fold_rows]
        summary.append(
            {
                "pas_label": config[0][0], "pas_descriptor": config[0],
                "pdp_alpha": config[1], "iterations": config[2],
                "deltas": deltas, "mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)),
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
    (ROOT / "phase9_anchor_joint_channel_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    run()
