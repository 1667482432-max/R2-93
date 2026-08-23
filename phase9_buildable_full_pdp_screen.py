from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import r2_pipeline as rp
from phase8_anchor_retained_pas_screen import mapped_anchors
from phase8_anchor_retained_pdp_screen import official_geometry
from phase9_anchor_full_pdp_screen import normalize


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
ANCHOR_BASE_K = (4, 8)
INTERPOLATION_K = (4, 8)
ALPHAS = (0.025, 0.05, 0.075, 0.10)


@torch.no_grad()
def descriptor(channel: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rows = []
    for start in range(0, len(indices), 4):
        x = torch.as_tensor(
            np.asarray(channel[indices[start:start + 4]]).copy(), device=DEVICE
        )
        pdp = torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2
        pdp /= torch.linalg.vector_norm(pdp, dim=-1, keepdim=True).clamp_min(1e-30)
        rows.append(pdp.cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def run() -> None:
    pos, channel, energy = rp.load_data()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    official_counts, actual_fraction, actual_counts = official_geometry(pos, energy, test_pos)
    keys = list(itertools.product(ANCHOR_BASE_K, INTERPOLATION_K, ALPHAS))
    fold_scores = {key: [] for key in keys}
    baselines = []
    epsilon = 1e-4 / 192
    valid = energy > 0
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        anchors = mapped_anchors(pos, val, labels, actual_fraction, official_counts)
        query_local = np.setdiff1d(np.arange(len(val)), anchors)
        base_all = np.load(ROOT / f"phase9_anchor_fold{fold}_base_pdp_band192.npy", mmap_mode="r")
        truth_all = np.load(ROOT / f"phase9_anchor_fold{fold}_truth_pdp_band192.npy", mmap_mode="r")
        mask = valid.copy()
        mask[val] = False
        external_train = np.flatnonzero(mask)
        distance_external, local_external = cKDTree(pos[external_train, :2]).query(
            pos[val[anchors], :2], k=max(ANCHOR_BASE_K)
        )
        external_indices = external_train[local_external]
        unique_indices, inverse = np.unique(external_indices, return_inverse=True)
        external_pdp = descriptor(channel, unique_indices)[inverse].reshape(
            len(anchors), max(ANCHOR_BASE_K), 256, 4, 192
        )
        weights = np.asarray(
            [official_counts[int(group)] / np.sum(labels[query_local] == group)
             for group in labels[query_local]], dtype=np.float64
        )
        denominator = float(weights.sum()) * 256 * 4
        baseline_sum = 0.0
        accumulator = {key: 0.0 for key in keys}
        for group in np.unique(labels[query_local]):
            rows = np.flatnonzero(labels[query_local] == group)
            local_query = query_local[rows]
            group_anchor_local = np.flatnonzero(labels[anchors] == group)
            base = np.asarray(base_all[local_query])
            truth = np.asarray(truth_all[local_query])
            base_cos = np.sum(base * truth, axis=-1) / np.maximum(
                np.linalg.norm(base, axis=-1) * np.linalg.norm(truth, axis=-1), 1e-30
            )
            weighted_base = float(np.sum(base_cos * weights[rows, None, None]))
            baseline_sum += weighted_base
            if len(group_anchor_local) == 0:
                for key in keys:
                    accumulator[key] += weighted_base
                continue
            interp_max = len(group_anchor_local)
            interp_distance, interp_local = cKDTree(
                pos[val[anchors[group_anchor_local]], :2]
            ).query(pos[val[local_query], :2], k=interp_max)
            interp_distance = np.asarray(interp_distance)
            interp_local = np.asarray(interp_local)
            if interp_max == 1:
                interp_distance = interp_distance[:, None]
                interp_local = interp_local[:, None]
            anchor_truth = np.asarray(truth_all[anchors[group_anchor_local]])
            for anchor_base_k in ANCHOR_BASE_K:
                distance = distance_external[group_anchor_local, :anchor_base_k]
                scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
                anchor_weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** 3.0
                anchor_weight /= anchor_weight.sum(1, keepdims=True)
                anchor_base = np.einsum(
                    "nk,nkaus->naus", anchor_weight, external_pdp[group_anchor_local, :anchor_base_k],
                    optimize=True,
                )
                anchor_log = np.clip(
                    np.log((anchor_truth + epsilon) / (anchor_base + epsilon)), -2.0, 2.0
                )
                for interpolation_k in INTERPOLATION_K:
                    k = min(interpolation_k, interp_max)
                    distance = interp_distance[:, :k]
                    scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1.0)
                    interp_weight = 1.0 / np.maximum(distance + 0.10 * scale, 0.25) ** 0.5
                    interp_weight /= interp_weight.sum(1, keepdims=True)
                    predicted_log = np.einsum(
                        "rk,rkaus->raus", interp_weight, anchor_log[interp_local[:, :k]],
                        optimize=True,
                    )
                    for alpha in ALPHAS:
                        desired = normalize(base * np.exp(alpha * predicted_log))
                        cosine = np.sum(desired * truth, axis=-1) / np.maximum(
                            np.linalg.norm(desired, axis=-1) * np.linalg.norm(truth, axis=-1),
                            1e-30,
                        )
                        accumulator[anchor_base_k, interpolation_k, alpha] += float(
                            np.sum(cosine * weights[rows, None, None])
                        )
        baseline = baseline_sum / denominator
        baselines.append(baseline)
        for key in keys:
            fold_scores[key].append(accumulator[key] / denominator)
        print(json.dumps({"stage": "fold", "fold": fold, "baseline": baseline}), flush=True)

    baseline_array = np.asarray(baselines)
    results = []
    for key in keys:
        delta = 0.4 * (np.asarray(fold_scores[key]) - baseline_array)
        results.append(
            {
                "anchor_base_k": key[0], "interpolation_k": key[1], "alpha": key[2],
                "score_proxy_deltas": delta.tolist(), "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()), "lcb": float(delta.mean() - delta.std()),
            }
        )
    results.sort(key=lambda row: (row["lcb"], row["mean_delta"]), reverse=True)
    output = {
        "official_test_counts": official_counts,
        "actual_anchor_counts": actual_counts,
        "results": results,
    }
    (ROOT / "phase9_buildable_full_pdp_screen.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "top": results}), flush=True)


if __name__ == "__main__":
    run()
