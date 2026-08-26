from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

import r2_pipeline as rp
os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import BANDS, build_cache
from matched_phase5_tree_descriptor import model_features


ROOT = Path(__file__).resolve().parent
SHIFT_SCALE = float(os.environ.get("R2_CANONICAL_SHIFT", "5.0"))
SHIFT_COMPONENT = int(os.environ.get("R2_CANONICAL_COMPONENT", "1"))
V_SHIFT_SCALE = float(os.environ.get("R2_CANONICAL_V_SHIFT", "0.0"))
V_SHIFT_COMPONENT = int(os.environ.get("R2_CANONICAL_V_COMPONENT", "1"))
FRACTIONAL = os.environ.get("R2_CANONICAL_FRACTIONAL", "0") == "1"
LABEL = (
    f"c{SHIFT_COMPONENT}_k{str(SHIFT_SCALE).replace('.', 'p').replace('-', 'm')}"
    + (
        f"_vc{V_SHIFT_COMPONENT}_k{str(V_SHIFT_SCALE).replace('.', 'p').replace('-', 'm')}"
        if V_SHIFT_SCALE != 0.0
        else ""
    )
    + ("_frac" if FRACTIONAL else "")
)


def normalize_bs(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)


def direction_component(pos: np.ndarray, component: int = SHIFT_COMPONENT) -> np.ndarray:
    xy = pos[:, :2]
    side = xy[:, 1] > 0
    bs = np.where(side[:, None], np.array([52.0, 35.0]), np.array([-18.413, -65.881]))
    relative = xy - bs
    unit = relative / np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-9)
    return unit[:, component]


def shifts(pos: np.ndarray) -> np.ndarray:
    value = SHIFT_SCALE * direction_component(pos)
    return value.astype(np.float32) if FRACTIONAL else np.rint(value).astype(np.int64)


def v_shifts(pos: np.ndarray) -> np.ndarray:
    value = V_SHIFT_SCALE * direction_component(pos, V_SHIFT_COMPONENT)
    return value.astype(np.float32) if FRACTIONAL else np.rint(value).astype(np.int64)


def roll_hv(rows: np.ndarray, h_amount: np.ndarray, v_amount: np.ndarray) -> np.ndarray:
    shaped = np.asarray(rows).reshape(len(rows), 2, 16, 8, 4, BANDS)
    output = np.empty_like(shaped)
    for index, (h_shift, v_shift) in enumerate(zip(h_amount, v_amount)):
        if FRACTIONAL:
            h_frequency = np.fft.fftfreq(16)
            v_frequency = np.fft.fftfreq(8)
            phase = (
                np.exp(-2j * np.pi * h_frequency * float(h_shift))[:, None]
                * np.exp(-2j * np.pi * v_frequency * float(v_shift))[None, :]
            )
            transformed = np.fft.fft2(shaped[index], axes=(1, 2))
            output[index] = np.fft.ifft2(
                transformed * phase[None, :, :, None, None], axes=(1, 2)
            ).real.astype(output.dtype)
        else:
            output[index] = np.roll(
                shaped[index], (int(h_shift), int(v_shift)), axis=(1, 2)
            )
    return output.reshape(rows.shape)


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    features = model_features(
        np.vstack((pos, test_pos)), np.load(ROOT / "los_map_features.npy")
    )
    valid = np.flatnonzero(energy > 0)
    test_groups = rp.official_island_labels(test_pos)
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    all_h_shift = shifts(pos)
    all_v_shift = v_shifts(pos)
    fold_data = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        if FRACTIONAL:
            canonical = roll_hv(
                np.sqrt(np.maximum(np.asarray(target[train]), 0)),
                -all_h_shift[train],
                -all_v_shift[train],
            )
            transformed = canonical.reshape(len(train), -1)
        else:
            canonical = roll_hv(
                np.asarray(target[train]), -all_h_shift[train], -all_v_shift[train]
            )
            transformed = np.sqrt(np.maximum(canonical, 0)).reshape(len(train), -1)
        pca = PCA(n_components=160, svd_solver="randomized", random_state=29100 + fold)
        coefficient = pca.fit_transform(transformed)

        tree_model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=29200 + fold,
        )
        tree_model.fit(features[train], coefficient)
        tree = pca.inverse_transform(tree_model.predict(features[val]))
        tree = tree.reshape(len(val), 256, 4, BANDS)
        tree = np.maximum(
            roll_hv(tree, all_h_shift[val], all_v_shift[val]), 0
        ) ** 2
        tree = normalize_bs(tree).astype(np.float32)
        np.save(ROOT / f"matched_tree_pas_canonical_{LABEL}_fold{fold}.npy", tree)

        spatial_scale = np.std(pos[train, :2], axis=0)
        rbf_model = RBFInterpolator(
            pos[train, :2] / spatial_scale,
            coefficient,
            neighbors=100,
            smoothing=0.1,
            kernel="linear",
            degree=0,
        )
        rbf = pca.inverse_transform(rbf_model(pos[val, :2] / spatial_scale))
        rbf = rbf.reshape(len(val), 256, 4, BANDS)
        rbf = np.maximum(
            roll_hv(rbf, all_h_shift[val], all_v_shift[val]), 0
        ) ** 2
        rbf = normalize_bs(rbf).astype(np.float32)
        np.save(ROOT / f"matched_rbf_pas_canonical_{LABEL}_fold{fold}.npy", rbf)

        base = np.load(ROOT / f"matched_phase5_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
        truth = np.asarray(target[val])
        weights = np.asarray([counts[int(group)] / np.sum(labels == group) for group in labels])
        fold_data.append((base, tree, rbf, truth, weights))
        print(json.dumps({"stage": "fold", "fold": fold, "label": LABEL}), flush=True)

    baseline = []
    for base, _, _, truth, weights in fold_data:
        cosine = np.sum(base * truth, axis=1)
        baseline.append(float(np.sum(weights * cosine.mean((1, 2))) / weights.sum()))
    baseline = np.asarray(baseline)
    summary = []
    for tree_weight in np.arange(0.0, 0.51, 0.05):
        for rbf_weight in np.arange(0.0, 0.51, 0.05):
            tree_weight = round(float(tree_weight), 2)
            rbf_weight = round(float(rbf_weight), 2)
            if tree_weight + rbf_weight == 0 or tree_weight + rbf_weight > 0.8:
                continue
            values = []
            for base, tree, rbf, truth, weights in fold_data:
                prediction = normalize_bs(
                    (1.0 - tree_weight - rbf_weight) * base
                    + tree_weight * tree
                    + rbf_weight * rbf
                )
                cosine = np.sum(prediction * truth, axis=1)
                values.append(float(np.sum(weights * cosine.mean((1, 2))) / weights.sum()))
            delta = np.asarray(values) - baseline
            summary.append(
                {
                    "tree_weight": tree_weight,
                    "rbf_weight": rbf_weight,
                    "deltas": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "min_delta": float(delta.min()),
                    "lcb": float(delta.mean() - 0.75 * delta.std()),
                }
            )
    summary.sort(key=lambda item: item["lcb"], reverse=True)
    (ROOT / f"matched_phase6_pas_canonical_{LABEL}.json").write_text(
        json.dumps({"label": LABEL, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"label": LABEL, "top": summary[:30]}), flush=True)


if __name__ == "__main__":
    run()
