from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import r2_pipeline as rp
from matched_local_spectral_calibration import (
    DEVICE,
    apply_per_sample,
    residual_profiles,
)
from matched_spectral_calibration import torch_score


ROOT = Path(__file__).resolve().parent
FEATURES = ("xy", "geo", "geo_map", "geo_spec", "map_spec", "geo_map_spec")
NEIGHBORS = (5, 10, 20)
BLENDS = (0.05, 0.10, 0.20)
AXES = ("pas", "pdp", "both")


@torch.no_grad()
def prediction_summaries(pred: np.ndarray) -> np.ndarray:
    """Compact label-free spectral context for residual matching."""
    rows = []
    for start in range(0, len(pred), 4):
        stop = min(start + 4, len(pred))
        x = torch.as_tensor(np.asarray(pred[start:stop]).copy(), device=DEVICE)
        pas = torch.abs(rp.bs_fft_torch(x)) ** 2
        pas = pas.reshape(len(x), 2, 16, 8, 4, 192).sum((1, 4, 5))
        pas = pas.reshape(len(x), 8, 2, 4, 2).mean((2, 4)).flatten(1)
        pas /= torch.linalg.vector_norm(pas, dim=1, keepdim=True).clamp_min(1e-30)
        pdp = torch.abs(torch.fft.fft(x, dim=-1, norm="ortho")) ** 2
        pdp = pdp.sum((1, 2)).reshape(len(x), 24, 8).mean(2)
        pdp /= torch.linalg.vector_norm(pdp, dim=1, keepdim=True).clamp_min(1e-30)
        rows.append(torch.cat((pas, pdp), 1).cpu().numpy().astype(np.float32))
    return np.concatenate(rows)


def robust_block(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center = np.median(reference, axis=0)
    scale = np.quantile(reference, 0.75, axis=0) - np.quantile(reference, 0.25, axis=0)
    scale = np.maximum(scale, np.std(reference, axis=0) * 0.25)
    scale = np.maximum(scale, 1e-6)
    return ((values - center) / scale) / np.sqrt(max(1, values.shape[1]))


def feature_matrix(
    name: str,
    train_rows: list[dict],
    query: dict,
    train_mask: list[np.ndarray],
    query_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_xy = np.concatenate([row["xy"][mask] for row, mask in zip(train_rows, train_mask)])
    query_xy = query["xy"][query_mask]
    side = bool(np.mean(query_xy[:, 1]) > 0)
    bs = np.array([52.0, 35.0]) if side else np.array([-18.413, -65.881])
    train_rel = train_xy - bs
    query_rel = query_xy - bs
    train_geo = np.column_stack(
        (
            train_xy,
            np.linalg.norm(train_rel, axis=1),
            train_rel[:, 0] / np.maximum(np.linalg.norm(train_rel, axis=1), 1e-6),
            train_rel[:, 1] / np.maximum(np.linalg.norm(train_rel, axis=1), 1e-6),
        )
    )
    query_geo = np.column_stack(
        (
            query_xy,
            np.linalg.norm(query_rel, axis=1),
            query_rel[:, 0] / np.maximum(np.linalg.norm(query_rel, axis=1), 1e-6),
            query_rel[:, 1] / np.maximum(np.linalg.norm(query_rel, axis=1), 1e-6),
        )
    )
    train_map = np.concatenate([row["map"][mask] for row, mask in zip(train_rows, train_mask)])
    query_map = query["map"][query_mask]
    train_spec = np.concatenate([row["spec"][mask] for row, mask in zip(train_rows, train_mask)])
    query_spec = query["spec"][query_mask]

    blocks = {
        "xy": ((train_xy, query_xy),),
        "geo": ((train_geo, query_geo),),
        "geo_map": ((train_geo, query_geo), (train_map, query_map)),
        "geo_spec": ((train_geo, query_geo), (train_spec, query_spec)),
        "map_spec": ((train_map, query_map), (train_spec, query_spec)),
        "geo_map_spec": (
            (train_geo, query_geo),
            (train_map, query_map),
            (train_spec, query_spec),
        ),
    }[name]
    train_parts = []
    query_parts = []
    for train_block, query_block in blocks:
        reference = train_block
        train_parts.append(robust_block(train_block, reference))
        query_parts.append(robust_block(query_block, reference))
    return np.column_stack(train_parts), np.column_stack(query_parts)


def predict_ratios(
    train_feature: np.ndarray,
    query_feature: np.ndarray,
    train_pas: np.ndarray,
    train_pdp: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(neighbors, len(train_feature))
    distance, index = cKDTree(train_feature).query(query_feature, k=k)
    distance = np.asarray(distance)
    index = np.asarray(index)
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]
    scale = max(float(np.median(distance[:, -1])), 0.1)
    weight = 1.0 / (distance + 0.25 * scale) ** 2
    weight /= weight.sum(1, keepdims=True)
    pas = np.sum(train_pas[index] * weight[:, :, None, None], axis=1)
    pdp = np.sum(train_pdp[index] * weight[:, :, None, None], axis=1)
    return np.exp(pas).astype(np.float32), np.exp(pdp).astype(np.float32)


def run() -> None:
    pos, channel, _ = rp.load_data()
    map_features = np.load(ROOT / "los_map_features.npy")[: len(pos)]
    folds = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        pred = np.load(ROOT / f"matched_phase4_full_fold{fold}.npy", mmap_mode="r")
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
                "map": map_features[val],
                "spec": prediction_summaries(pred),
                "pas": pas,
                "pdp": pdp,
            }
        )
        print(json.dumps({"stage": "features", "fold": fold}), flush=True)

    groups = sorted(set(np.concatenate([row["labels"] for row in folds]).tolist()))
    baselines = {}
    for row in folds:
        for group in groups:
            mask = row["labels"] == group
            p = torch.as_tensor(np.asarray(row["pred"][mask]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(row["target"][mask]).copy(), device=DEVICE)
            baselines[row["fold"], group] = torch_score(p, t)

    records = []
    for feature_name in FEATURES:
        for neighbors in NEIGHBORS:
            for row in folds:
                train_rows = [other for other in folds if other["fold"] != row["fold"]]
                for group in groups:
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
                    pas_ratio, pdp_ratio = predict_ratios(
                        train_feature,
                        query_feature,
                        train_pas,
                        train_pdp,
                        neighbors,
                    )
                    p = torch.as_tensor(
                        np.asarray(row["pred"][query_mask]).copy(), device=DEVICE
                    )
                    t = torch.as_tensor(
                        np.asarray(row["target"][query_mask]).copy(), device=DEVICE
                    )
                    for blend in BLENDS:
                        for axis in AXES:
                            active_pas = (
                                pas_ratio if axis in {"pas", "both"} else np.ones_like(pas_ratio)
                            )
                            active_pdp = (
                                pdp_ratio if axis in {"pdp", "both"} else np.ones_like(pdp_ratio)
                            )
                            corrected = apply_per_sample(
                                p, active_pas, active_pdp, blend
                            )
                            score = torch_score(corrected, t)
                            records.append(
                                {
                                    "feature": feature_name,
                                    "neighbors": neighbors,
                                    "blend": blend,
                                    "axis": axis,
                                    "fold": row["fold"],
                                    "group": group,
                                    "delta": score["score"]
                                    - baselines[row["fold"], group]["score"],
                                    **score,
                                }
                            )
            print(
                json.dumps(
                    {"stage": "config", "feature": feature_name, "neighbors": neighbors}
                ),
                flush=True,
            )

    summary = []
    for feature_name in FEATURES:
        for neighbors in NEIGHBORS:
            for blend in BLENDS:
                for axis in AXES:
                    for group in groups:
                        selected = [
                            row
                            for row in records
                            if row["feature"] == feature_name
                            and row["neighbors"] == neighbors
                            and row["blend"] == blend
                            and row["axis"] == axis
                            and row["group"] == group
                        ]
                        delta = np.asarray([row["delta"] for row in selected])
                        summary.append(
                            {
                                "feature": feature_name,
                                "neighbors": neighbors,
                                "blend": blend,
                                "axis": axis,
                                "group": group,
                                "deltas": delta.tolist(),
                                "mean_delta": float(delta.mean()),
                                "min_delta": float(delta.min()),
                                "positive": int(np.sum(delta > 0)),
                                "lcb": float(delta.mean() - 0.75 * delta.std()),
                            }
                        )
    summary.sort(key=lambda item: item["lcb"], reverse=True)
    safe = [item for item in summary if item["min_delta"] > 0]
    payload = {"records": records, "summary": summary, "safe": safe}
    (ROOT / "matched_phase5_feature_residual.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:25], "safe": safe[:25]}), flush=True)


if __name__ == "__main__":
    run()
