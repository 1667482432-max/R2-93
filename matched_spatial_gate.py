from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent


def build_current(fold: int, labels: np.ndarray) -> Path:
    path = ROOT / f"matched_phase_full_fold{fold}.npy"
    expected = (len(labels), 256, 4, 192)
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.complex64:
            return path
    base = np.load(ROOT / f"matched_pred_core_nog10_safe_fold{fold}.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.complex64, shape=expected)
    for start in range(0, len(base), 4):
        output[start:start + 4] = base[start:start + 4]
    paths = {
        0: f"matched_extended_g0_fold{fold}.npy",
        1: f"matched_extended_g1_fold{fold}.npy",
        3: f"matched_phase_g3_fold{fold}.npy",
        4: f"matched_phase_g4_fold{fold}.npy",
        5: f"matched_phase_g5_fold{fold}.npy",
        6: f"matched_phase_g6_fold{fold}.npy",
        7: f"matched_map_g7_fold{fold}.npy",
        8: f"matched_phase_g8_fold{fold}.npy",
        9: f"matched_phase_g9_fold{fold}.npy",
    }
    for group, name in paths.items():
        rows = np.flatnonzero(labels == group)
        output[rows] = np.load(ROOT / name, mmap_mode="r")
    output.flush()
    return path


def align_candidate(current: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    candidate = candidate.astype(np.complex64, copy=True)
    cross = np.sum(np.conj(candidate) * current, axis=(1, 2, 3), dtype=np.complex128)
    phase = cross / np.maximum(np.abs(cross), 1e-30)
    candidate *= phase[:, None, None, None].astype(np.complex64)
    return candidate


def cosine(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    return np.sum(a * b, axis=axis) / np.maximum(
        np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis), 1e-30
    )


def spectral_features_and_gain(current, candidate, target):
    rows = []
    gains = []
    components = []
    for start in range(0, len(target), 2):
        stop = min(start + 2, len(target))
        c = np.asarray(current[start:stop])
        d = align_candidate(c, np.asarray(candidate[start:stop]))
        t = np.asarray(target[start:stop])
        cp = np.abs(rp.bs_fft_numpy(c)) ** 2
        dp = np.abs(rp.bs_fft_numpy(d)) ** 2
        tp = np.abs(rp.bs_fft_numpy(t)) ** 2
        cd = np.abs(np.fft.fft(c, axis=-1, norm="ortho")) ** 2
        dd = np.abs(np.fft.fft(d, axis=-1, norm="ortho")) ** 2
        td = np.abs(np.fft.fft(t, axis=-1, norm="ortho")) ** 2
        cv_pas = cosine(cp, dp, 1).mean((1, 2))
        cv_pdp = cosine(cd, dd, -1).mean((1, 2))
        c1 = cosine(cp, tp, 1).mean((1, 2))
        d1 = cosine(dp, tp, 1).mean((1, 2))
        c2 = cosine(cd, td, -1).mean((1, 2))
        d2 = cosine(dd, td, -1).mean((1, 2))
        ce = np.sum(np.abs(c - t) ** 2, axis=(1, 2, 3), dtype=np.float64)
        de = np.sum(np.abs(d - t) ** 2, axis=(1, 2, 3), dtype=np.float64)
        te = np.sum(np.abs(t) ** 2, axis=(1, 2, 3), dtype=np.float64)
        c3, d3 = ce / te, de / te
        gain = 0.4 * (d1 - c1) + 0.4 * (d2 - c2) + 0.2 * (
            1 / (1 + d3) - 1 / (1 + c3)
        )
        complex_cos = np.abs(np.sum(np.conj(c) * d, axis=(1, 2, 3))) / np.maximum(
            np.linalg.norm(c.reshape(len(c), -1), axis=1) *
            np.linalg.norm(d.reshape(len(d), -1), axis=1), 1e-30
        )
        energy_ratio = np.log(np.maximum(
            np.sum(np.abs(d) ** 2, axis=(1, 2, 3), dtype=np.float64) /
            np.maximum(np.sum(np.abs(c) ** 2, axis=(1, 2, 3), dtype=np.float64), 1e-30),
            1e-30,
        ))
        difference = (
            np.sum(np.abs(c - d) ** 2, axis=(1, 2, 3), dtype=np.float64) /
            np.maximum(np.sum(np.abs(c) ** 2, axis=(1, 2, 3), dtype=np.float64), 1e-30)
        )
        def distribution_stats(power, axis):
            probability = power / np.maximum(power.sum(axis=axis, keepdims=True), 1e-30)
            entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-30)), axis=axis)
            entropy /= np.log(power.shape[axis])
            peak = probability.max(axis=axis)
            return entropy.mean(tuple(range(1, entropy.ndim))), peak.mean(tuple(range(1, peak.ndim)))
        cpe, cpp = distribution_stats(cp, 1)
        dpe, dpp = distribution_stats(dp, 1)
        cde, cdp = distribution_stats(cd, -1)
        dde, ddp = distribution_stats(dd, -1)
        feature = np.column_stack([
            cv_pas, cv_pdp, complex_cos, energy_ratio, difference,
            cpe, dpe, cpp, dpp, cde, dde, cdp, ddp,
        ])
        rows.append(feature)
        gains.append(gain)
        components.append(np.column_stack([c1, d1, c2, d2, ce, de, te]))
    return np.concatenate(rows), np.concatenate(gains), np.concatenate(components)


def geometry_features(pos, energy, fold: int, val: np.ndarray, labels: np.ndarray):
    valid = np.flatnonzero(energy > 0)
    train = np.setdiff1d(valid, val, assume_unique=False)
    query = pos[val, :2]
    train_xy = pos[train, :2]
    distance, index = cKDTree(train_xy).query(query, k=32)
    selected = distance[:, [0, 1, 3, 7, 15, 31]]
    delta = train_xy[index[:, :16]] - query[:, None, :]
    mean_delta = delta.mean(1)
    std_delta = delta.std(1)
    covariance = (delta[:, :, 0] * delta[:, :, 1]).mean(1)
    train_groups = rp.official_island_labels(pos[train])
    group_distance = np.zeros((len(val), 5), np.float64)
    group_relative = np.zeros((len(val), 4), np.float64)
    for group in np.unique(labels):
        mask = labels == group
        local = train_xy[train_groups == group]
        if len(local) < 2:
            local = train_xy
        k = min(16, len(local))
        local_distance, _ = cKDTree(local).query(query[mask], k=k)
        local_distance = np.atleast_2d(local_distance)
        picks = np.minimum(np.asarray([0, 1, 3, 7, 15]), k - 1)
        group_distance[mask] = local_distance[:, picks]
        lo, hi = np.quantile(local, [0.05, 0.95], axis=0)
        scale = np.maximum(hi - lo, 1.0)
        relative = (query[mask] - lo) / scale
        group_relative[mask, :2] = relative
        group_relative[mask, 2:] = (query[mask] - local.mean(0)) / np.maximum(local.std(0), 1.0)
    maps = np.load(ROOT / "los_map_features.npy", mmap_mode="r")[val]
    one_hot = np.eye(11, dtype=np.float64)[labels]
    return np.column_stack([
        query, selected, mean_delta, std_delta, covariance,
        group_distance, group_relative, maps, one_hot,
    ])


class GatedView:
    def __init__(self, current, candidate, weight):
        self.current = current
        self.candidate = candidate
        self.weight = np.asarray(weight, np.float32)

    def __len__(self):
        return len(self.current)

    def __getitem__(self, item):
        c = np.asarray(self.current[item])
        d = np.asarray(self.candidate[item])
        scalar = np.asarray(self.weight[item])
        single = c.ndim == 3
        if single:
            c, d, scalar = c[None], d[None], scalar[None]
        d = align_candidate(c, d)
        out = (1 - scalar[:, None, None, None]) * c + scalar[:, None, None, None] * d
        return out[0] if single else out


def model_configs():
    for leaf in (5, 10, 20, 40):
        yield f"extra_leaf{leaf}", lambda leaf=leaf: ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=leaf, max_features=0.75,
            n_jobs=-1, random_state=1667,
        )
    for leaf in (10, 20, 40):
        yield f"hist_leaf{leaf}", lambda leaf=leaf: HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=leaf, l2_regularization=0.05, random_state=1667,
        )


def run() -> None:
    pos, channel, energy = rp.load_data()
    folds = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        current_path = build_current(fold, labels)
        current = np.load(current_path, mmap_mode="r")
        candidate = np.load(ROOT / f"matched_pred_v5_rejected_fold{fold}.npy", mmap_mode="r")
        spectral, gain, components = spectral_features_and_gain(
            current, candidate, channel[val]
        )
        geometry = geometry_features(pos, energy, fold, val, labels)
        x = np.column_stack([geometry, spectral])
        folds.append({
            "fold": fold, "val": val, "labels": labels, "x": x, "gain": gain,
            "current": current_path, "components": components,
        })
        print(json.dumps({
            "fold": fold, "n": len(val), "features": x.shape[1],
            "gain_mean": float(gain.mean()), "gain_positive": float(np.mean(gain > 0)),
        }), flush=True)
    np.savez_compressed(ROOT / "matched_spatial_gate_features.npz", **{
        f"x{row['fold']}": row["x"] for row in folds
    }, **{f"gain{row['fold']}": row["gain"] for row in folds}, **{
        f"components{row['fold']}": row["components"] for row in folds
    })

    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    test_counts = dict(zip(*np.unique(test_groups, return_counts=True)))
    mappings = {
        "hard0": lambda z: (z > 0).astype(np.float32),
        "hard002": lambda z: (z > 0.002).astype(np.float32),
        "hard005": lambda z: (z > 0.005).astype(np.float32),
        "hard010": lambda z: (z > 0.010).astype(np.float32),
    }
    results = []
    for model_name, factory in model_configs():
        prediction = []
        for holdout in range(5):
            train_x = np.concatenate([row["x"] for row in folds if row["fold"] != holdout])
            train_y = np.concatenate([row["gain"] for row in folds if row["fold"] != holdout])
            model = factory()
            model.fit(train_x, train_y)
            prediction.append(model.predict(folds[holdout]["x"]))
        for mapping_name, mapping in mappings.items():
            scores = []
            baselines = []
            selections = []
            for row, estimated in zip(folds, prediction):
                weights = mapping(estimated)
                labels = row["labels"]
                sample_weights = np.asarray([
                    test_counts.get(int(group), 0) / max(1, np.sum(labels == group))
                    for group in labels
                ])
                c1, d1, c2, d2, ce, de, te = row["components"].T
                def aggregate(gate):
                    pas = np.average(np.where(gate > 0, d1, c1), weights=sample_weights)
                    pdp = np.average(np.where(gate > 0, d2, c2), weights=sample_weights)
                    error = np.sum(sample_weights * np.where(gate > 0, de, ce))
                    target_energy = np.sum(sample_weights * te)
                    nmse = error / target_energy
                    return 0.4 * pas + 0.4 * pdp + 0.2 / (1 + nmse)
                baselines.append(aggregate(np.zeros_like(weights)))
                scores.append(aggregate(weights))
                selections.append(float(weights.mean()))
            delta = np.asarray(scores) - baselines
            result = {
                "model": model_name, "mapping": mapping_name,
                "scores": scores, "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()), "min_delta": float(delta.min()),
                "selections": selections,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: row["mean_delta"], reverse=True)
    (ROOT / "matched_spatial_gate_validation.json").write_text(
        json.dumps({"results": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best": results[0]}), flush=True)


if __name__ == "__main__":
    run()
