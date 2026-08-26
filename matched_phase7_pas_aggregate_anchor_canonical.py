from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import build_cache
from matched_phase6_pas_fitted_canonical import (
    direction,
    fit_coefficients as fit_h_coefficients,
    h_moment,
    shifts as h_shifts,
)
from matched_phase6_pas_fitted_vertical import (
    fit_coefficients as fit_v_coefficients,
    shifts as v_shifts,
    v_moment,
)
from matched_phase7_pas_aggregate_canonical import (
    aggregate,
    correct,
    normalize_profile,
    point_cosine,
    roll_profile,
)


ROOT = Path(__file__).resolve().parent
VARIANTS = ("plain", "h", "hv")
METHODS = ("nearest", "idw1", "idw2", "idw3", "mean")
SCALES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def run() -> None:
    pos, _, energy = rp.load_data()
    valid_mask = energy > 0
    valid = np.flatnonzero(valid_mask)
    target = build_cache()
    aggregate_target = aggregate(target)
    unit, side = direction(pos)
    horizontal_moment = h_moment(target, valid)
    vertical_moment = v_moment(target, valid)
    diagnostics = json.loads((ROOT / "matched_rect_split_diagnostics.json").read_text())
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        h_coefficient = fit_h_coefficients(
            unit, side, valid, train, horizontal_moment, 59800 + 10 * fold
        )
        v_coefficient = fit_v_coefficients(
            unit, side, valid, train, vertical_moment, 59900 + 10 * fold
        )
        horizontal_shift = h_shifts(unit, side, h_coefficient)
        vertical_shift = v_shifts(unit, side, v_coefficient)
        base = np.asarray(
            np.load(
                ROOT / f"matched_phase6_milestone_physics_pas_band24_fold{fold}.npy",
                mmap_mode="r",
            )
        )
        truth = np.asarray(target[val])
        baseline_point = point_cosine(base, truth)
        weights = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels]
        )
        candidates = {
            (variant, method): aggregate(base).copy()
            for variant in VARIANTS
            for method in METHODS
        }
        anchor_counts = {}
        for group in counts:
            item = next(
                row
                for row in diagnostics
                if row["fold"] == fold and row["group"] == int(group)
            )
            lo, hi = np.asarray(item["box_lo"]), np.asarray(item["box_hi"])
            inside = valid_mask & np.all(pos[:, :2] >= lo, axis=1) & np.all(
                pos[:, :2] <= hi, axis=1
            )
            anchors = np.flatnonzero(inside & ~np.isin(np.arange(len(pos)), val))
            mask = labels == group
            query = val[mask]
            anchor_counts[str(group)] = int(len(anchors))
            if not len(anchors):
                continue
            distance, local = cKDTree(pos[anchors, :2]).query(
                pos[query, :2], k=len(anchors)
            )
            if len(anchors) == 1:
                distance, local = distance[:, None], local[:, None]
            for variant in VARIANTS:
                h_anchor = (
                    -horizontal_shift[anchors]
                    if variant in ("h", "hv")
                    else np.zeros(len(anchors), int)
                )
                v_anchor = (
                    -vertical_shift[anchors]
                    if variant == "hv"
                    else np.zeros(len(anchors), int)
                )
                source = roll_profile(aggregate_target[anchors], h_anchor, v_anchor)
                h_query = (
                    horizontal_shift[query]
                    if variant in ("h", "hv")
                    else np.zeros(len(query), int)
                )
                v_query = (
                    vertical_shift[query]
                    if variant == "hv"
                    else np.zeros(len(query), int)
                )
                values = {"nearest": source[local[:, 0]]}
                for power in (1, 2, 3):
                    local_weight = 1.0 / np.maximum(distance, 0.5) ** power
                    local_weight /= local_weight.sum(1, keepdims=True)
                    values[f"idw{power}"] = np.einsum(
                        "nk,nkp->np", local_weight, source[local], optimize=True
                    )
                values["mean"] = np.broadcast_to(source.mean(0), (len(query), 256))
                for method, value in values.items():
                    candidates[variant, method][mask] = normalize_profile(
                        np.maximum(roll_profile(value, h_query, v_query), 0)
                    )
        for (variant, method), desired in candidates.items():
            for scale in SCALES:
                prediction = correct(base, desired, scale)
                delta_point = point_cosine(prediction, truth) - baseline_point
                records.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "method": method,
                        "scale": scale,
                        "delta": float(np.sum(weights * delta_point) / weights.sum()),
                        "groups": {
                            str(group): float(delta_point[labels == group].mean())
                            for group in counts
                        },
                    }
                )
        print(
            json.dumps({"stage": "fold", "fold": fold, "anchors": anchor_counts}),
            flush=True,
        )

    summary = []
    group_summary = []
    for variant, method, scale in itertools.product(VARIANTS, METHODS, SCALES):
        selected = [
            row
            for row in records
            if row["variant"] == variant
            and row["method"] == method
            and row["scale"] == scale
        ]
        delta = np.asarray([row["delta"] for row in selected])
        common = {"variant": variant, "method": method, "scale": scale}
        summary.append(
            {
                **common,
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
        for group in counts:
            local = np.asarray([row["groups"][str(group)] for row in selected])
            group_summary.append(
                {
                    **common,
                    "group": int(group),
                    "deltas": local.tolist(),
                    "mean_delta": float(local.mean()),
                    "min_delta": float(local.min()),
                    "lcb": float(local.mean() - 0.75 * local.std()),
                }
            )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    group_summary.sort(key=lambda row: row["lcb"], reverse=True)
    safe_groups = [row for row in group_summary if row["min_delta"] > 0]
    payload = {"summary": summary, "safe_groups": safe_groups, "groups": group_summary}
    (ROOT / "matched_phase7_pas_aggregate_anchor_canonical.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(
        json.dumps({"top": summary[:30], "safe_groups": safe_groups[:80]}),
        flush=True,
    )


if __name__ == "__main__":
    run()
