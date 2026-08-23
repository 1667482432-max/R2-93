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
from phase8_anchor_retained_pas_screen import horizontal_shifts, mapped_anchors, normalize
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_anchor_joint_channel_validation import normalize_last, project_joint
from phase9_buildable_full_pdp_screen import descriptor as pdp_descriptor
from phase9_buildable_residual_gate_pas_screen import external_residual


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
PAS_DESCRIPTORS = (
    ("robust", 0.15, 0.75, "residual_then_local"),
    ("balanced", 0.10, 0.75, "local_then_residual"),
    ("aggressive", 0.15, 0.75, "local_then_residual"),
)
PDP_ALPHAS = (0.0, 0.025)
ITERATIONS = 12
CONFIGS = tuple(itertools.product(PAS_DESCRIPTORS, PDP_ALPHAS))


def external_pdp_residual(
    pos: np.ndarray,
    channel: np.ndarray,
    val: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    query_local: np.ndarray,
    external_train: np.ndarray,
    truth_all: np.ndarray,
) -> np.ndarray:
    distance_external, local_external = cKDTree(pos[external_train, :2]).query(
        pos[val[anchors], :2], k=8
    )
    external_indices = external_train[local_external]
    unique_indices, inverse = np.unique(external_indices, return_inverse=True)
    external_pdp = pdp_descriptor(channel, unique_indices)[inverse].reshape(
        len(anchors), 8, 256, 4, 192
    )
    distance = distance_external
    scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
    weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** 3.0
    weight /= weight.sum(1, keepdims=True)
    anchor_base = np.einsum("nk,nkaus->naus", weight, external_pdp, optimize=True)
    anchor_truth = np.asarray(truth_all[anchors])
    epsilon = 1e-4 / 192
    anchor_log = np.clip(
        np.log((anchor_truth + epsilon) / (anchor_base + epsilon)), -2.0, 2.0
    )
    output = np.zeros((len(query_local), 256, 4, 192), dtype=np.float32)
    for group in np.unique(labels[query_local]):
        rows = np.flatnonzero(labels[query_local] == group)
        group_anchor_local = np.flatnonzero(labels[anchors] == group)
        if len(group_anchor_local) == 0:
            continue
        k = min(4, len(group_anchor_local))
        distance, local = cKDTree(pos[val[anchors[group_anchor_local]], :2]).query(
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
        output[rows] = np.einsum(
            "rk,rkaus->raus", weight, anchor_log[group_anchor_local[local]], optimize=True
        )
    return output


@torch.no_grad()
def run() -> None:
    folds, pos, channel, energy, official_counts, actual_counts = prepare_folds()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    _, actual_fraction, _ = official_geometry(pos, energy, test_pos)
    target = np.load(ROOT / "pas_ue_band24_descriptors.npy", mmap_mode="r")
    shifts = horizontal_shifts(pos)
    valid = energy > 0
    for fold, row in enumerate(folds):
        anchors = mapped_anchors(pos, row["val"], row["labels"], actual_fraction, official_counts)
        mask = valid.copy()
        mask[row["val"]] = False
        external_train = np.flatnonzero(mask)
        row["pas_residual"] = external_residual(
            pos, row["val"], row["labels"], anchors, row["query_local"],
            target, external_train, shifts, "none",
        )
        base_pdp_all = np.load(
            ROOT / f"phase9_anchor_fold{fold}_base_pdp_band192.npy", mmap_mode="r"
        )
        truth_pdp_all = np.load(
            ROOT / f"phase9_anchor_fold{fold}_truth_pdp_band192.npy", mmap_mode="r"
        )
        row["base_pdp"] = np.asarray(base_pdp_all[row["query_local"]])
        row["pdp_residual"] = external_pdp_residual(
            pos, channel, row["val"], row["labels"], anchors, row["query_local"],
            external_train, truth_pdp_all,
        )
        print(json.dumps({"stage": "residual", "fold": fold}), flush=True)

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
            _, residual_alpha, local_scale, order = descriptor
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
            alpha: normalize_last(
                row["base_pdp"] * np.exp(alpha * row["pdp_residual"])
            ).astype(np.float32)
            for alpha in PDP_ALPHAS
        }
        labels = row["labels"]
        query_local = row["query_local"]
        query = row["query"]
        prediction = np.load(ROOT / f"matched_phase6_full_fold{fold}.npy", mmap_mode="r")
        accumulator = np.zeros((1 + len(CONFIGS), 6), np.float64)
        for start in range(0, len(query), 2):
            stop = min(start + 2, len(query))
            local_query = query_local[start:stop]
            p = torch.as_tensor(np.asarray(prediction[local_query]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(channel[query[start:stop]]).copy(), device=DEVICE)
            weights = torch.as_tensor(row["weights"][start:stop].astype(np.float32), device=DEVICE)
            truth_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            truth_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
            update_scores(accumulator, None, 0, p, t, truth_pas, truth_pdp, weights, labels[local_query])
            base_pas_band = torch.as_tensor(row["base"][start:stop].copy(), device=DEVICE)
            for index, (pas_descriptor, pdp_alpha) in enumerate(CONFIGS, 1):
                value = project_joint(
                    p, base_pas_band,
                    torch.as_tensor(desired_pas[pas_descriptor][start:stop], device=DEVICE),
                    torch.as_tensor(desired_pdp[pdp_alpha][start:stop], device=DEVICE),
                    ITERATIONS,
                )
                update_scores(
                    accumulator, None, index, value, t, truth_pas, truth_pdp,
                    weights, labels[local_query],
                )
        baseline = components(accumulator[0])
        rows = []
        for index, config in enumerate(CONFIGS, 1):
            value = components(accumulator[index])
            rows.append(
                {
                    "pas_label": config[0][0], "pas_descriptor": config[0],
                    "pdp_alpha": config[1], **value,
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
                "pdp_alpha": config[1], "iterations": ITERATIONS,
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
    (ROOT / "phase9_buildable_joint_channel_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    run()
