from __future__ import annotations

import gc
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import BANDS, base_descriptor, build_cache, normalize_last
from matched_phase5_tree_descriptor import model_features
from matched_phase6_pas_canonical_descriptor import roll_hv
from matched_phase6_pas_fitted_canonical import (
    direction,
    fit_coefficients as fit_h_coefficients,
    h_moment,
    roll_h,
    shifts as h_shifts,
)
from matched_phase6_pas_fitted_vertical import (
    fit_coefficients as fit_v_coefficients,
    roll_v,
    shifts as v_shifts,
    v_moment,
)
from matched_phase6_pas_mlp_descriptor import fit_predict as fit_old_mlp, fourier_features
from matched_phase6_pas_rich_gate import descriptor_stats, similarity_stats
from matched_phase6_pas_rich_mlp_descriptor import fit_predict as fit_rich_mlp


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BASE_PATH = ROOT / "Round2_Test_Channel_matched_phase5_delta1642.npy"
OUTPUT = ROOT / "Round2_Test_Channel_matched_phase6_delta2053.npy"
MANIFEST = ROOT / "matched_phase6_manifest.json"
N_TRAIN = 4000
BAND_WIDTH = 192 // BANDS
SCALE = 1.25
PDP_STRENGTH = 1.5
ITERATIONS = 12

GATE_CONFIGS = {
    4: {"threshold": 0.01, "leaf": 10, "alpha": 0.50, "probability": 0.60},
    5: {"threshold": 0.003, "leaf": 2, "alpha": 0.30, "probability": 0.45},
}
KNN_CONFIGS = {
    3: {"dims": 16, "map_weight": 4.0, "alpha": 0.0375},
    7: {"dims": 16, "map_weight": 4.0, "alpha": 0.0375},
}
GROUPWISE = {
    1: (0.05, 0.15, 0.50),
    3: (0.00, 0.00, 1.00),
    4: (0.15, 0.00, 1.50),
    5: (0.00, 0.00, 1.00),
    7: (0.05, 0.10, 2.00),
    9: (0.00, 0.025, 1.50),
}
ENHANCED_RULES = {
    0: (11, 6.5402, "low", 0.30),
    3: (2, 2.9594, "low", 0.30),
    5: (13, 18.1151, "low", 0.20),
    7: (15, 43.645328521728516, "high", 0.75),
    10: (10, 6.0715, "low", 0.05),
}
MILESTONE_RULES = {
    0: ("h_leaf2", 9, -0.6488235473632791, "high", 0.20),
    1: ("vertical", 6, 18.14078559875488, "low", 0.15),
    5: ("canonical5", 0, 56.46230506896973, "low", 0.075),
    6: ("h_leaf3_mf5", 8, 0.14137563109397888, "high", 1.00),
    7: ("canonical5", 15, 43.645328521728516, "high", 0.40),
    9: ("vertical", 10, 6.596559524536133, "high", 0.10),
    10: ("rich_mlp", 8, -2.2051992416381836, "low", 0.50),
}


def log(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}), flush=True)


def normalize_bs(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def normalized_torch(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=dim, keepdim=True).clamp_min(1e-30)


def save_prediction(name: str, value: np.ndarray) -> np.ndarray:
    path = ROOT / name
    value = normalize_bs(value).astype(np.float32)
    np.save(path, value)
    log("descriptor_saved", name=path.name, shape=list(value.shape))
    return value


def cached(name: str) -> np.ndarray | None:
    path = ROOT / name
    if not path.exists():
        return None
    value = np.load(path, mmap_mode="r")
    expected = (500, 256, 4, BANDS)
    if value.shape != expected or value.dtype != np.float32:
        return None
    log("descriptor_cached", name=name)
    return np.asarray(value)


def fit_plain_pca(target: np.ndarray, train: np.ndarray, components: int, seed: int, order: str):
    rows = np.asarray(target[train])
    if order == "transposed":
        rows = rows.transpose(0, 2, 3, 1)
    transformed = np.sqrt(np.maximum(rows, 0)).reshape(len(train), -1)
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        n_oversamples=20,
        iterated_power=4,
        random_state=seed,
    )
    coefficient = pca.fit_transform(transformed)
    del rows, transformed
    gc.collect()
    return pca, coefficient


def reconstruct_plain(pca: PCA, coefficient: np.ndarray, order: str) -> np.ndarray:
    prediction = pca.inverse_transform(coefficient)
    if order == "transposed":
        prediction = np.maximum(prediction.reshape(len(coefficient), 4, BANDS, 256), 0) ** 2
        prediction = normalize_last(prediction).transpose(0, 3, 1, 2)
    else:
        prediction = np.maximum(prediction.reshape(len(coefficient), 256, 4, BANDS), 0) ** 2
        prediction = normalize_bs(prediction)
    return prediction.astype(np.float32)


def build_rich_tree(
    target: np.ndarray, valid: np.ndarray, test_index: np.ndarray, rich_features: np.ndarray
) -> np.ndarray:
    name = f"matched_rich_tree_pas_band{BANDS}_test.npy"
    value = cached(name)
    if value is not None:
        return value
    pca, coefficient = fit_plain_pca(target, valid, 192, 30105, "direct")
    model = ExtraTreesRegressor(
        n_estimators=700,
        min_samples_leaf=3,
        max_features=0.65,
        n_jobs=-1,
        random_state=30205,
    )
    model.fit(rich_features[valid], coefficient)
    prediction = reconstruct_plain(pca, model.predict(rich_features[test_index]), "direct")
    del pca, coefficient, model
    gc.collect()
    return save_prediction(name, prediction)


def build_old_mlp(
    target: np.ndarray, valid: np.ndarray, test_index: np.ndarray, features: np.ndarray
) -> np.ndarray:
    name = f"matched_mlp_pas_band{BANDS}_test.npy"
    value = cached(name)
    if value is not None:
        return value
    pca, coefficient = fit_plain_pca(target, valid, 160, 27105, "transposed")
    predicted = fit_old_mlp(features, coefficient, valid, test_index, 27205)
    prediction = reconstruct_plain(pca, predicted, "transposed")
    del pca, coefficient, predicted
    gc.collect()
    return save_prediction(name, prediction)


def build_rich_mlp(
    target: np.ndarray, valid: np.ndarray, test_index: np.ndarray, features: np.ndarray
) -> np.ndarray:
    name = f"matched_rich_mlp_pas_band{BANDS}_test.npy"
    value = cached(name)
    if value is not None:
        return value
    pca, coefficient = fit_plain_pca(target, valid, 192, 38105, "transposed")
    predicted = fit_rich_mlp(features, coefficient, valid, test_index, 38205)
    prediction = reconstruct_plain(pca, predicted, "transposed")
    del pca, coefficient, predicted
    gc.collect()
    return save_prediction(name, prediction)


def build_canonical5(
    target: np.ndarray,
    valid: np.ndarray,
    test_index: np.ndarray,
    all_pos: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    name = "matched_tree_pas_canonical_c1_k5p0_test.npy"
    value = cached(name)
    if value is not None:
        return value
    xy = all_pos[:, :2]
    side = xy[:, 1] > 0
    bs = np.where(side[:, None], np.array([52.0, 35.0]), np.array([-18.413, -65.881]))
    relative = xy - bs
    unit = relative / np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-9)
    h_shift = np.rint(5.0 * unit[:, 1]).astype(np.int64)
    v_shift = np.zeros(len(all_pos), dtype=np.int64)
    canonical = roll_hv(np.asarray(target[valid]), -h_shift[valid], -v_shift[valid])
    transformed = np.sqrt(np.maximum(canonical, 0)).reshape(len(valid), -1)
    pca = PCA(n_components=160, svd_solver="randomized", random_state=29105)
    coefficient = pca.fit_transform(transformed)
    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=29205,
    )
    model.fit(features[valid], coefficient)
    prediction = pca.inverse_transform(model.predict(features[test_index]))
    prediction = prediction.reshape(len(test_index), 256, 4, BANDS)
    prediction = np.maximum(
        roll_hv(prediction, h_shift[test_index], v_shift[test_index]), 0
    ) ** 2
    del canonical, transformed, pca, coefficient, model
    gc.collect()
    return save_prediction(name, prediction)


def build_horizontal(
    target: np.ndarray,
    valid: np.ndarray,
    test_index: np.ndarray,
    all_pos: np.ndarray,
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    names = {
        "horizontal": "matched_fitted_canonical_leaf8_mf8_pas_band24_test.npy",
        "h_leaf2": "matched_fitted_canonical_leaf2_mf8_pas_band24_test.npy",
        "h_leaf3_mf5": "matched_fitted_canonical_leaf3_mf5_pas_band24_test.npy",
    }
    existing = {key: cached(name) for key, name in names.items()}
    if all(value is not None for value in existing.values()):
        return {key: value for key, value in existing.items() if value is not None}
    unit, side = direction(all_pos)
    moment = h_moment(target, valid)
    coefficients = fit_h_coefficients(unit, side, valid, valid, moment, 41150)
    shift = h_shifts(unit, side, coefficients)
    canonical = roll_h(np.asarray(target[valid]), -shift[valid])
    transformed = np.sqrt(np.maximum(canonical, 0)).reshape(len(valid), -1)
    pca = PCA(
        n_components=160,
        svd_solver="randomized",
        n_oversamples=20,
        iterated_power=4,
        random_state=41205,
    )
    coefficient = pca.fit_transform(transformed)
    configs = {
        "horizontal": (8, 0.8),
        "h_leaf2": (2, 0.8),
        "h_leaf3_mf5": (3, 0.5),
    }
    result: dict[str, np.ndarray] = {}
    for key, (leaf, max_features) in configs.items():
        if existing[key] is not None:
            result[key] = existing[key]
            continue
        model = ExtraTreesRegressor(
            n_estimators=320,
            min_samples_leaf=leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=41300 + 500 + 10 * leaf + int(10 * max_features),
        )
        model.fit(features[valid], coefficient)
        prediction = pca.inverse_transform(model.predict(features[test_index]))
        prediction = np.maximum(
            roll_h(prediction.reshape(len(test_index), 256, 4, BANDS), shift[test_index]), 0
        ) ** 2
        result[key] = save_prediction(names[key], prediction)
    log(
        "horizontal_coefficients",
        coefficients={str(key): value.tolist() for key, value in coefficients.items()},
    )
    del canonical, transformed, pca, coefficient
    gc.collect()
    return result


def build_vertical(
    target: np.ndarray,
    valid: np.ndarray,
    test_index: np.ndarray,
    all_pos: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    name = "matched_fitted_vertical_pas_band24_test.npy"
    value = cached(name)
    if value is not None:
        return value
    unit, side = direction(all_pos)
    moment = v_moment(target, valid)
    coefficients = fit_v_coefficients(unit, side, valid, valid, moment, 42150)
    shift = v_shifts(unit, side, coefficients)
    canonical = roll_v(np.asarray(target[valid]), -shift[valid])
    transformed = np.sqrt(np.maximum(canonical, 0)).reshape(len(valid), -1)
    pca = PCA(
        n_components=160,
        svd_solver="randomized",
        n_oversamples=20,
        iterated_power=4,
        random_state=42205,
    )
    coefficient = pca.fit_transform(transformed)
    model = ExtraTreesRegressor(
        n_estimators=420,
        min_samples_leaf=8,
        max_features=0.8,
        n_jobs=-1,
        random_state=42305,
    )
    model.fit(features[valid], coefficient)
    prediction = pca.inverse_transform(model.predict(features[test_index]))
    prediction = np.maximum(
        roll_v(prediction.reshape(len(test_index), 256, 4, BANDS), shift[test_index]), 0
    ) ** 2
    log(
        "vertical_coefficients",
        coefficients={str(key): value.tolist() for key, value in coefficients.items()},
    )
    del canonical, transformed, pca, coefficient, model
    gc.collect()
    return save_prediction(name, prediction)


def geometry_features_test(
    pos: np.ndarray,
    test_pos: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    map_features: np.ndarray,
) -> np.ndarray:
    query = test_pos[:, :2]
    train_xy = pos[valid, :2]
    distance, index = cKDTree(train_xy).query(query, k=32)
    selected = distance[:, [0, 1, 3, 7, 15, 31]]
    delta = train_xy[index[:, :16]] - query[:, None, :]
    mean_delta = delta.mean(1)
    std_delta = delta.std(1)
    covariance = (delta[:, :, 0] * delta[:, :, 1]).mean(1)
    train_groups = rp.official_island_labels(pos[valid])
    group_distance = np.zeros((len(test_pos), 5), np.float64)
    group_relative = np.zeros((len(test_pos), 4), np.float64)
    for group in np.unique(labels):
        mask = labels == group
        local = train_xy[train_groups == group]
        if len(local) < 2:
            local = train_xy
        k = min(16, len(local))
        local_distance, _ = cKDTree(local).query(query[mask], k=k)
        local_distance = np.asarray(local_distance)
        if local_distance.ndim == 1:
            local_distance = local_distance[:, None]
        picks = np.minimum(np.asarray([0, 1, 3, 7, 15]), k - 1)
        group_distance[mask] = local_distance[:, picks]
        lo, hi = np.quantile(local, [0.05, 0.95], axis=0)
        local_scale = np.maximum(hi - lo, 1.0)
        relative = (query[mask] - lo) / local_scale
        group_relative[mask, :2] = relative
        group_relative[mask, 2:] = (
            (query[mask] - local.mean(0)) / np.maximum(local.std(0), 1.0)
        )
    one_hot = np.eye(11, dtype=np.float64)[labels]
    return np.column_stack(
        [
            query,
            selected,
            mean_delta,
            std_delta,
            covariance,
            group_distance,
            group_relative,
            map_features[N_TRAIN:],
            one_hot,
        ]
    )


def build_gate_features(
    geometry: np.ndarray,
    rich_map: np.ndarray,
    components: list[np.ndarray],
) -> np.ndarray:
    blocks = [geometry, rich_map[N_TRAIN:]]
    blocks.extend(descriptor_stats(component) for component in components)
    for left, right in itertools.combinations(range(len(components)), 2):
        blocks.append(similarity_stats(components[left], components[right]))
    value = np.column_stack(blocks).astype(np.float32)
    expected = np.load(ROOT / "matched_phase6_pas_rich_gate_features.npz")["x0"].shape[1]
    if value.shape != (500, expected):
        raise RuntimeError(f"gate feature mismatch: {value.shape}, expected (500, {expected})")
    return value


def full_gate_alphas(test_x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cache = np.load(ROOT / "matched_phase6_pas_rich_gate_features.npz")
    alpha = np.zeros(len(labels), dtype=np.float32)
    for group, config in GATE_CONFIGS.items():
        train_x = []
        train_y = []
        for fold in range(5):
            fold_labels = cache[f"labels{fold}"]
            mask = fold_labels == group
            train_x.append(cache[f"x{fold}"][mask])
            train_y.append(cache[f"oracle_gain{fold}"][mask] > config["threshold"])
        train_y_all = np.concatenate(train_y)
        model = ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=config["leaf"],
            max_features=0.65,
            class_weight="balanced",
            n_jobs=-1,
            random_state=32200 + 100 * group + 10 * config["leaf"] + 5,
        )
        model.fit(np.concatenate(train_x), train_y_all)
        mask = labels == group
        probability = model.predict_proba(test_x[mask])[:, list(model.classes_).index(True)]
        chosen = probability >= config["probability"]
        alpha[mask] = config["alpha"] * chosen
        log("test_gate", group=group, selected=int(chosen.sum()), total=int(mask.sum()))
    return alpha


def map_latent_features(
    pos: np.ndarray, test_pos: np.ndarray, energy: np.ndarray, rich_map: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    all_pos = np.vstack((pos, test_pos))
    valid = np.flatnonzero(energy > 0)
    rich = rich_map.astype(np.float64)
    center = np.median(rich[valid], axis=0)
    scale = np.quantile(rich[valid], 0.75, axis=0) - np.quantile(rich[valid], 0.25, axis=0)
    scale = np.maximum(scale, np.std(rich[valid], axis=0) * 0.1)
    scale = np.maximum(scale, 1e-3)
    standardized = np.clip((rich - center) / scale, -10, 10)
    latent = PCA(n_components=32, whiten=True, random_state=33100).fit_transform(standardized)
    xy_center = pos[valid, :2].mean(0)
    xy_scale = pos[valid, :2].std(0)
    xy = (all_pos[:, :2] - xy_center) / xy_scale
    return xy, latent


def build_desired(
    pos: np.ndarray,
    test_pos: np.ndarray,
    energy: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    geometry: np.ndarray,
    test_x: np.ndarray,
    rich_map: np.ndarray,
    components: dict[str, np.ndarray],
) -> np.ndarray:
    base = components["base"]
    rich = components["rich_tree"]
    safe = base.copy()
    gate_alpha = full_gate_alphas(test_x, labels)
    gate_mask = gate_alpha > 0
    if np.any(gate_mask):
        a = gate_alpha[gate_mask, None, None, None]
        safe[gate_mask] = normalize_bs((1.0 - a) * base[gate_mask] + a * rich[gate_mask])

    xy, latent = map_latent_features(pos, test_pos, energy, rich_map)
    all_pos = np.vstack((pos, test_pos))
    test_index = np.arange(N_TRAIN, N_TRAIN + len(test_pos))
    for group, config in KNN_CONFIGS.items():
        mask = labels == group
        feature = np.column_stack((xy, config["map_weight"] * latent[:, : config["dims"]]))
        distance, local = cKDTree(feature[valid]).query(feature[test_index[mask]], k=32)
        indices = valid[local]
        weight = 1.0 / np.maximum(distance, 0.05)
        weight /= weight.sum(1, keepdims=True)
        candidate = np.empty((mask.sum(), 256, 4, BANDS), dtype=np.float32)
        for row in range(mask.sum()):
            candidate[row] = np.einsum(
                "k,kaub->aub", weight[row], np.asarray(target[indices[row]]), optimize=True
            )
        candidate = normalize_bs(candidate).astype(np.float32)
        a = config["alpha"]
        safe[mask] = normalize_bs((1.0 - a) * base[mask] + a * candidate)
        log("test_knn", group=group, selected=int(mask.sum()))

    horizontal = components["horizontal"]
    vertical = components["vertical"]
    desired = base.copy()
    for group, (h_alpha, v_alpha, safe_scale) in GROUPWISE.items():
        mask = labels == group
        desired[mask] = normalize_bs(
            (1.0 - h_alpha - v_alpha - safe_scale) * base[mask]
            + h_alpha * horizontal[mask]
            + v_alpha * vertical[mask]
            + safe_scale * safe[mask]
        )
    group6 = (labels == 6) & (geometry[:, 8] > 1.3770753145217889)
    desired[group6] = normalize_bs(0.5 * desired[group6] + 0.5 * rich[group6])
    log("groupwise", group6_rows=int(group6.sum()))

    for group, (feature, threshold, rule_direction, alpha) in ENHANCED_RULES.items():
        mask = labels == group
        mask &= geometry[:, feature] <= threshold if rule_direction == "low" else geometry[:, feature] >= threshold
        desired[mask] = normalize_bs((1.0 - alpha) * desired[mask] + alpha * rich[mask])
        log("enhanced_rule", group=group, selected=int(mask.sum()))

    for group, (name, feature, threshold, rule_direction, alpha) in MILESTONE_RULES.items():
        candidate = components[name]
        mask = labels == group
        mask &= geometry[:, feature] <= threshold if rule_direction == "low" else geometry[:, feature] >= threshold
        desired[mask] = normalize_bs((1.0 - alpha) * desired[mask] + alpha * candidate[mask])
        log("milestone_rule", group=group, candidate=name, selected=int(mask.sum()))
    path = ROOT / "matched_phase6_milestone_physics_pas_band24_test.npy"
    np.save(path, desired.astype(np.float32))
    log("desired_saved", name=path.name)
    return desired.astype(np.float32)


@torch.no_grad()
def project(desired: np.ndarray) -> None:
    base = np.load(BASE_PATH, mmap_mode="r")
    base_band = base_descriptor(base)
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), 4):
        stop = min(start + 4, len(base))
        p = torch.as_tensor(np.asarray(base[start:stop]).copy(), device=DEVICE)
        base_pas_complex = rp.bs_fft_torch(p)
        base_pas = torch.abs(base_pas_complex) ** 2
        band = torch.as_tensor(np.asarray(base_band[start:stop]).copy(), device=DEVICE)
        desired_band = torch.as_tensor(np.asarray(desired[start:stop]).copy(), device=DEVICE)
        target_band = normalized_torch((1.0 - SCALE) * band + SCALE * desired_band, 1)
        epsilon = 1e-3 / band.shape[1]
        ratio = ((target_band + epsilon) / (band + epsilon)).clamp(0.25, 4.0)
        target_pas = base_pas * ratio.repeat_interleave(BAND_WIDTH, dim=3)
        base_pdp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
        x = rp.bs_ifft_torch(
            base_pas_complex * torch.sqrt(target_pas / base_pas.clamp_min(1e-30))
        )
        for _ in range(ITERATIONS):
            z = torch.fft.fft(x, dim=-1, norm="ortho")
            correction = torch.sqrt(base_pdp).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
            x = torch.fft.ifft(z * correction.pow(PDP_STRENGTH), dim=-1, norm="ortho")
            z = rp.bs_fft_torch(x)
            correction = torch.sqrt(target_pas).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
            x = rp.bs_ifft_torch(z * correction)
        output[start:stop] = x.cpu().numpy().astype(np.complex64)
        if stop % 50 == 0 or stop == len(base):
            output.flush()
            log("project", done=stop)
    del output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_output(valid_count: int, outlier_count: int, selected_rows: dict[str, int]) -> dict:
    check = np.load(OUTPUT, mmap_mode="r")
    finite = True
    nonzero = True
    for start in range(0, len(check), 8):
        block = np.asarray(check[start : start + 8])
        finite &= bool(np.isfinite(block).all())
        nonzero &= bool(np.all(np.sum(np.abs(block), axis=(1, 2, 3)) > 0))
    validation_path = ROOT / "matched_phase6_milestone_physics_channel_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))["summary"]
    selected = next(
        row
        for row in validation
        if row["scale"] == SCALE
        and row["pdp_strength"] == PDP_STRENGTH
        and row["iterations"] == ITERATIONS
    )
    phase5_delta = 0.016416780366500593
    manifest = {
        "output": OUTPUT.name,
        "shape": list(check.shape),
        "dtype": str(check.dtype),
        "bytes": OUTPUT.stat().st_size,
        "finite": finite,
        "all_test_rows_nonzero": nonzero,
        "sha256": sha256(OUTPUT),
        "validation": selected,
        "delta_vs_phase5": selected["mean_delta"],
        "cumulative_delta_vs_matched_v3": phase5_delta + selected["mean_delta"],
        "training_rows_used": valid_count,
        "zero_channel_outliers_removed": outlier_count,
        "test_rule_selection_counts": selected_rows,
        "model": {
            "bands": BANDS,
            "projection_scale": SCALE,
            "pdp_strength": PDP_STRENGTH,
            "iterations": ITERATIONS,
            "validation_split": "five disjoint official-like rectangular holdouts",
            "selection_policy": "positive on all five folds; robust LCB prioritized",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("validated", **manifest)
    if not finite or not nonzero:
        raise RuntimeError("submission contains non-finite or all-zero rows")
    return manifest


def run() -> None:
    pos, _, energy = rp.load_data()
    if len(pos) != N_TRAIN:
        raise RuntimeError(f"unexpected training count: {len(pos)}")
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    all_pos = np.vstack((pos, test_pos))
    test_index = np.arange(N_TRAIN, N_TRAIN + len(test_pos))
    valid = np.flatnonzero(energy > 0)
    target = build_cache()
    los_map = np.load(ROOT / "los_map_features.npy")
    rich_map = np.load(ROOT / "rich_map_features.npy").astype(np.float32)
    basic = model_features(all_pos, los_map)
    rich_tree_features = model_features(all_pos, rich_map)
    old_mlp_features = fourier_features(all_pos, los_map)
    rich_mlp_features = np.column_stack((old_mlp_features, rich_map)).astype(np.float32)
    labels = rp.official_island_labels(test_pos).astype(np.int64)
    log(
        "data",
        valid_rows=int(len(valid)),
        removed_zero_rows=int(len(pos) - len(valid)),
        test_groups={str(g): int(np.sum(labels == g)) for g in np.unique(labels)},
    )

    base = base_descriptor(np.load(BASE_PATH, mmap_mode="r")).astype(np.float32)
    tree = np.asarray(np.load(ROOT / f"matched_tree_pas_band{BANDS}_test.npy", mmap_mode="r"))
    rbf = np.asarray(np.load(ROOT / f"matched_rbf_pas_band{BANDS}_test.npy", mmap_mode="r"))
    rich_tree = build_rich_tree(target, valid, test_index, rich_tree_features)
    old_mlp = build_old_mlp(target, valid, test_index, old_mlp_features)
    canonical5 = build_canonical5(target, valid, test_index, all_pos, basic)
    horizontal = build_horizontal(target, valid, test_index, all_pos, basic)
    vertical = build_vertical(target, valid, test_index, all_pos, basic)
    rich_mlp = build_rich_mlp(target, valid, test_index, rich_mlp_features)

    geometry = geometry_features_test(pos, test_pos, valid, labels, los_map)
    test_x = build_gate_features(
        geometry, rich_map, [base, rich_tree, tree, rbf, old_mlp, canonical5]
    )
    np.save(ROOT / "matched_phase6_pas_rich_gate_features_test.npy", test_x)
    components = {
        "base": base,
        "rich_tree": rich_tree,
        "horizontal": horizontal["horizontal"],
        "h_leaf2": horizontal["h_leaf2"],
        "h_leaf3_mf5": horizontal["h_leaf3_mf5"],
        "vertical": vertical,
        "canonical5": canonical5,
        "rich_mlp": rich_mlp,
    }
    desired = build_desired(
        pos,
        test_pos,
        energy,
        valid,
        target,
        labels,
        geometry,
        test_x,
        rich_map,
        components,
    )
    selected_rows = {
        str(group): int(np.sum(labels == group)) for group in np.unique(labels)
    }
    project(desired)
    validate_output(len(valid), len(pos) - len(valid), selected_rows)


if __name__ == "__main__":
    run()
