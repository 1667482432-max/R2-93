from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent
FOLDS = 5
SEED = 20260820


def inside_box(xy: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.all(xy >= lo, axis=1) & np.all(xy <= hi, axis=1)


def unit_mean(rows: np.ndarray) -> np.ndarray:
    value = rows.mean(0)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def angle_distance(a: float, b: float) -> float:
    return abs(float(np.arctan2(np.sin(a - b), np.cos(a - b))))


def distance_signature(query: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    k = min(16, len(anchors))
    distance = np.atleast_2d(cKDTree(anchors).query(query, k=k)[0])
    if len(query) == 1 and distance.shape[0] != 1:
        distance = distance.T
    columns = [0, min(3, k - 1), min(7, k - 1), min(15, k - 1)]
    return np.concatenate([
        np.quantile(distance[:, column], (0.1, 0.5, 0.9)) for column in columns
    ])


def run() -> None:
    pos, _, energy = rp.load_data()
    xy = pos[:, :2]
    valid = energy > 0
    valid_ids = np.flatnonzero(valid)
    test = np.load(ROOT / "Round2_Test_Pos.npy")
    # Keep the pipeline's exact group assignment. Two official boxes overlap;
    # the later box owns the overlap, so raw DBSCAN labels alone give the wrong
    # group populations (5/43 instead of the actual 2/46 for groups 2/6).
    raw_labels = DBSCAN(eps=10, min_samples=3).fit_predict(test[:, :2])
    labels = rp.official_island_labels(test)
    train_labels = rp.official_island_labels(pos)
    map_features = np.load(ROOT / "los_map_features.npy")
    descriptors = np.load(ROOT / "channel_descriptors.npz")
    pas, pdp = descriptors["pas"], descriptors["pdp"]
    log_energy = descriptors["log_energy"]

    # Robust scales make the map terms comparable without allowing a single
    # ray-count coordinate to dominate candidate selection.
    map_train = map_features[: len(pos)]
    map_med = np.median(map_train[valid], axis=0)
    map_iqr = np.quantile(map_train[valid], 0.75, axis=0) - np.quantile(
        map_train[valid], 0.25, axis=0
    )
    map_iqr = np.maximum(map_iqr, 1.0)

    specs = []
    official_mask = np.zeros(len(pos), bool)
    for group in sorted(np.unique(labels)):
        tq = test[labels == group]
        raw_group = test[raw_labels == group]
        lo, hi = raw_group[:, :2].min(0), raw_group[:, :2].max(0)
        anchors = np.flatnonzero(valid & (train_labels == group))
        official_mask |= inside_box(xy, lo - 5.0, hi + 5.0)
        side = bool(np.mean(tq[:, 1]) > 0)
        bs = np.array([52.0, 35.0]) if side else np.array([-18.413, -65.881])
        center = (lo + hi) / 2
        relevant = slice(4, 8) if side else slice(0, 4)
        target_map = (np.median(map_features[len(pos) :][labels == group, relevant], axis=0)
                      - map_med[relevant]) / map_iqr[relevant]
        target_signature = distance_signature(tq[:, :2], xy[valid])
        anchor_fraction = len(anchors) / (len(anchors) + len(tq))
        specs.append({
            "group": int(group), "size": hi - lo, "center": center,
            "side": side, "bs": bs, "query_n": len(tq),
            "anchor_n": len(anchors), "anchor_fraction": anchor_fraction,
            "official_anchors": anchors, "target_map": target_map,
            "target_signature": target_signature,
            "target_radius": float(np.linalg.norm(center - bs)),
            "target_angle": float(np.arctan2(*(center - bs)[::-1])),
            "target_pas": unit_mean(pas[anchors]) if len(anchors) else None,
            "target_pdp": unit_mean(pdp[anchors]) if len(anchors) else None,
            "target_log_energy": float(np.median(log_energy[anchors])) if len(anchors) else None,
        })

    rng = np.random.default_rng(SEED)
    candidates = valid_ids[~official_mask[valid_ids]]
    used_global = np.zeros(len(pos), bool)
    diagnostics = []

    for fold in range(FOLDS):
        selected: list[int] = []
        selected_groups: list[int] = []
        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        # Hard, sparse official islands get first choice of matched rectangles.
        order = sorted(specs, key=lambda spec: (spec["anchor_fraction"], -spec["query_n"]))
        for spec in order:
            pool = candidates[((pos[candidates, 1] > 0) == spec["side"])]
            rng.shuffle(pool)
            proposals = []
            seen_boxes: set[tuple[float, ...]] = set()
            for center_id in pool[:3200]:
                center = xy[center_id]
                lo, hi = center - spec["size"] / 2, center + spec["size"] / 2
                key = tuple(np.round(np.r_[lo, hi], 1))
                if key in seen_boxes:
                    continue
                seen_boxes.add(key)
                if any(np.all(hi + 4 >= old_lo) and np.all(lo - 4 <= old_hi)
                       for old_lo, old_hi in boxes):
                    continue
                rows = np.flatnonzero(valid & ~used_global & inside_box(xy, lo, hi))
                min_val = min(8, max(2, spec["query_n"]))
                if len(rows) < min_val + (1 if spec["anchor_n"] else 0):
                    continue
                desired_anchor = int(round(len(rows) * spec["anchor_fraction"]))
                if spec["anchor_n"]:
                    desired_anchor = max(1, desired_anchor)
                desired_anchor = min(desired_anchor, max(0, len(rows) - min_val))
                feature_slice = slice(4, 8) if spec["side"] else slice(0, 4)
                candidate_map = ((np.median(map_train[rows, feature_slice], axis=0)
                                  - map_med[feature_slice])
                                 / map_iqr[feature_slice])
                map_penalty = float(np.mean(np.abs(candidate_map - spec["target_map"])))
                radius = float(np.linalg.norm(center - spec["bs"]))
                angle = float(np.arctan2(*(center - spec["bs"])[::-1]))
                geometry_penalty = abs(radius - spec["target_radius"]) / max(
                    spec["target_radius"], 25.0
                ) + 0.25 * angle_distance(angle, spec["target_angle"])
                channel_penalty = 0.0
                if spec["target_pas"] is not None:
                    reliability = min(1.0, spec["anchor_n"] / 8.0)
                    channel_penalty = reliability * (
                        1.0 - float(unit_mean(pas[rows]) @ spec["target_pas"])
                        + 1.0 - float(unit_mean(pdp[rows]) @ spec["target_pdp"])
                        + 0.08 * abs(float(np.median(log_energy[rows]))
                                     - spec["target_log_energy"])
                    )
                target_count = min(max(spec["query_n"] // 3, 12), 35)
                count_penalty = abs((len(rows) - desired_anchor) - target_count) / 30.0
                prescore = (0.8 * map_penalty + 0.45 * geometry_penalty
                            + 1.2 * channel_penalty + 0.15 * count_penalty)
                proposals.append((prescore, rows, lo, hi, desired_anchor,
                                  map_penalty, geometry_penalty, channel_penalty))

            if not proposals:
                raise RuntimeError(f"No candidate rectangle for group {spec['group']} fold {fold}")
            proposals.sort(key=lambda item: item[0])
            best = None
            for proposal in proposals[:32]:
                (prescore, rows, lo, hi, n_anchor, map_penalty,
                 geometry_penalty, channel_penalty) = proposal
                trials = 1 if n_anchor == 0 else 16
                for _ in range(trials):
                    anchors = (np.empty(0, np.int64) if n_anchor == 0
                               else rng.choice(rows, n_anchor, replace=False))
                    val = rows[~np.isin(rows, anchors)]
                    train_ids = valid_ids[~np.isin(valid_ids, val)]
                    signature = distance_signature(xy[val], xy[train_ids])
                    distance_penalty = float(np.mean(np.abs(
                        np.log1p(signature) - np.log1p(spec["target_signature"])
                    )))
                    score = prescore + 1.6 * distance_penalty
                    if best is None or score < best[0]:
                        best = (score, val, rows, lo, hi, signature, n_anchor,
                                map_penalty, geometry_penalty, channel_penalty,
                                distance_penalty)
            assert best is not None
            (score, val, rectangle_rows, lo, hi, signature, n_anchor,
             map_penalty, geometry_penalty, channel_penalty,
             distance_penalty) = best
            selected.extend(val.tolist())
            selected_groups.extend([spec["group"]] * len(val))
            boxes.append((lo, hi))
            # Exclude the entire local support from all later folds. This avoids
            # tuning on a label that serves as an anchor in another fold.
            used_global |= inside_box(xy, lo - 4.0, hi + 4.0)
            diagnostics.append({
                "fold": fold, "group": spec["group"], "score": score,
                "rectangle_n": int(len(rectangle_rows)), "val_n": int(len(val)),
                "kept_anchor_n": int(n_anchor), "official_query_n": spec["query_n"],
                "official_anchor_n": spec["anchor_n"],
                "map_penalty": map_penalty,
                "geometry_penalty": geometry_penalty,
                "channel_penalty": channel_penalty,
                "distance_penalty": distance_penalty,
                "target_distance_signature": spec["target_signature"].tolist(),
                "candidate_distance_signature": signature.tolist(),
                "box_lo": lo.tolist(), "box_hi": hi.tolist(),
            })
            print(json.dumps(diagnostics[-1]), flush=True)

        selected_array = np.asarray(selected, np.int64)
        group_array = np.asarray(selected_groups, np.int64)
        order_index = np.argsort(selected_array)
        selected_array, group_array = selected_array[order_index], group_array[order_index]
        np.save(ROOT / f"matched_rect_val_{fold}.npy", selected_array)
        np.save(ROOT / f"matched_rect_groups_{fold}.npy", group_array)
        print(json.dumps({
            "fold": fold, "val_n": int(len(selected_array)),
            "counts": {int(group): int(np.sum(group_array == group))
                       for group in np.unique(group_array)},
        }), flush=True)

    (ROOT / "matched_rect_split_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
