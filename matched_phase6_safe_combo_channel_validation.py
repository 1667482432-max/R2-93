from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import build_cache
from matched_phase5_tree_channel_validation import update_score


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
BANDS = 24
BAND_WIDTH = 192 // BANDS
K = 32
GATE_CONFIGS = {
    4: {"threshold": 0.01, "leaf": 10, "alpha": 0.50, "probability": 0.60},
    5: {"threshold": 0.003, "leaf": 2, "alpha": 0.30, "probability": 0.45},
}
KNN_CONFIGS = {
    3: {"dims": 16, "map_weight": 4.0, "alpha": 0.0375},
    7: {"dims": 16, "map_weight": 4.0, "alpha": 0.0375},
}
CONFIGS = tuple(
    (scale, pdp_strength, iterations)
    for scale in (0.75, 1.00, 1.25)
    for pdp_strength, iterations in ((1.0, 12), (1.5, 4), (1.5, 12), (1.5, 24))
)


def normalize_bs(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-30)


def normalized(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=dim, keepdim=True).clamp_min(1e-30)


def map_latent_features(pos: np.ndarray, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    all_pos = np.vstack((pos, test_pos))
    rich = np.load(ROOT / "rich_map_features.npy").astype(np.float64)
    valid = np.flatnonzero(energy > 0)
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


def gate_alphas(fold: int, labels: np.ndarray) -> np.ndarray:
    cache = np.load(ROOT / "matched_phase6_pas_rich_gate_features.npz")
    alpha = np.zeros(len(labels), dtype=np.float32)
    for group, config in GATE_CONFIGS.items():
        train_x = []
        train_y = []
        for other in range(5):
            if other == fold:
                continue
            other_labels = cache[f"labels{other}"]
            mask = other_labels == group
            train_x.append(cache[f"x{other}"][mask])
            train_y.append(cache[f"oracle_gain{other}"][mask] > config["threshold"])
        model = ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=config["leaf"],
            max_features=0.65,
            class_weight="balanced",
            n_jobs=-1,
            random_state=32200 + 100 * group + 10 * config["leaf"] + fold,
        )
        model.fit(np.concatenate(train_x), np.concatenate(train_y))
        mask = labels == group
        probability = model.predict_proba(cache[f"x{fold}"][mask])[:, 1]
        alpha[mask] = config["alpha"] * (probability >= config["probability"])
    return alpha


def build_desired_descriptors() -> None:
    pos, _, energy = rp.load_data()
    valid = np.flatnonzero(energy > 0)
    target = build_cache()
    xy, latent = map_latent_features(pos, energy)
    for fold in range(5):
        output_path = ROOT / f"matched_phase6_safe_combo_pas_band24_fold{fold}.npy"
        if output_path.exists():
            print(json.dumps({"stage": "descriptor_cached", "fold": fold}), flush=True)
            continue
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        base = np.asarray(
            np.load(ROOT / f"matched_phase5_pas_band24_fold{fold}.npy", mmap_mode="r")
        )
        desired = base.copy()

        gate_alpha = gate_alphas(fold, labels)
        rich_candidate = np.load(
            ROOT / f"matched_rich_tree_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        gate_mask = gate_alpha > 0
        if np.any(gate_mask):
            a = gate_alpha[gate_mask, None, None, None]
            desired[gate_mask] = normalize_bs(
                (1.0 - a) * base[gate_mask] + a * np.asarray(rich_candidate)[gate_mask]
            )

        for group, config in KNN_CONFIGS.items():
            mask = labels == group
            feature = np.column_stack(
                (xy, config["map_weight"] * latent[:, : config["dims"]])
            )
            distance, local = cKDTree(feature[train]).query(feature[val[mask]], k=K)
            indices = train[local]
            weight = 1.0 / np.maximum(distance, 0.05)
            weight /= weight.sum(1, keepdims=True)
            candidate = np.empty((mask.sum(), 256, 4, BANDS), dtype=np.float32)
            for row in range(mask.sum()):
                candidate[row] = np.einsum(
                    "k,kaub->aub", weight[row], np.asarray(target[indices[row]]), optimize=True
                )
            candidate = normalize_bs(candidate).astype(np.float32)
            a = config["alpha"]
            desired[mask] = normalize_bs((1.0 - a) * base[mask] + a * candidate)

        np.save(output_path, desired.astype(np.float32))
        print(
            json.dumps(
                {
                    "stage": "descriptor",
                    "fold": fold,
                    "gate_rows": int(np.sum(gate_mask)),
                    "knn_rows": int(np.sum(np.isin(labels, list(KNN_CONFIGS)))),
                }
            ),
            flush=True,
        )


@torch.no_grad()
def run() -> None:
    build_desired_descriptors()
    _, channel, _ = rp.load_data()
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    fold_records = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        weights_np = np.asarray(
            [counts[int(group)] / np.sum(labels == group) for group in labels], dtype=np.float64
        )
        base = np.load(ROOT / f"matched_phase5_full_fold{fold}.npy", mmap_mode="r")
        desired = np.load(
            ROOT / f"matched_phase6_safe_combo_pas_band24_fold{fold}.npy", mmap_mode="r"
        )
        accumulator = np.zeros((1 + len(CONFIGS), 6), np.float64)
        for start in range(0, len(val), 4):
            stop = min(start + 4, len(val))
            p = torch.as_tensor(np.asarray(base[start:stop]).copy(), device=DEVICE)
            t = torch.as_tensor(np.asarray(channel[val[start:stop]]).copy(), device=DEVICE)
            w = torch.as_tensor(weights_np[start:stop], device=DEVICE)
            tt_pas = torch.abs(rp.bs_fft_torch(t)) ** 2
            tt_pdp = torch.abs(torch.fft.fft(t, dim=-1, norm="ortho")) ** 2
            base_pas_complex = rp.bs_fft_torch(p)
            base_pas = torch.abs(base_pas_complex) ** 2
            base_band = normalized(
                normalized(base_pas, 1)
                .reshape(len(p), 256, 4, BANDS, BAND_WIDTH)
                .mean(4),
                1,
            )
            desired_band = torch.as_tensor(
                np.asarray(desired[start:stop]).copy(), device=DEVICE
            )
            base_pdp = torch.abs(torch.fft.fft(p, dim=-1, norm="ortho")) ** 2
            update_score(accumulator, 0, p, t, tt_pas, tt_pdp, w)
            for config_index, (scale, pdp_strength, iterations) in enumerate(CONFIGS, 1):
                target_band = normalized((1.0 - scale) * base_band + scale * desired_band, 1)
                epsilon = 1e-3 / base_band.shape[1]
                ratio = ((target_band + epsilon) / (base_band + epsilon)).clamp(0.25, 4.0)
                target_pas = base_pas * ratio.repeat_interleave(BAND_WIDTH, dim=3)
                x = rp.bs_ifft_torch(
                    base_pas_complex * torch.sqrt(target_pas / base_pas.clamp_min(1e-30))
                )
                for _ in range(iterations):
                    z = torch.fft.fft(x, dim=-1, norm="ortho")
                    correction = torch.sqrt(base_pdp).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
                    x = torch.fft.ifft(z * correction.pow(pdp_strength), dim=-1, norm="ortho")
                    z = rp.bs_fft_torch(x)
                    correction = torch.sqrt(target_pas).clamp_min(1e-30) / torch.abs(z).clamp_min(1e-30)
                    x = rp.bs_ifft_torch(z * correction)
                update_score(accumulator, config_index, x, t, tt_pas, tt_pdp, w)
        rows = []
        for index, config in enumerate(((0.0, 0.0, 0),) + CONFIGS):
            c1 = accumulator[index, 0] / accumulator[index, 4]
            c2 = accumulator[index, 1] / accumulator[index, 5]
            c3 = accumulator[index, 2] / accumulator[index, 3]
            rows.append(
                {
                    "scale": config[0],
                    "pdp_strength": config[1],
                    "iterations": config[2],
                    "c1_pas": c1,
                    "c2_pdp": c2,
                    "c3_nmse": c3,
                    "score": 0.4 * c1 + 0.4 * c2 + 0.2 / (1.0 + c3),
                }
            )
        fold_records.append({"fold": fold, "rows": rows})
        print(
            json.dumps({"stage": "fold", "fold": fold, "best": max(rows, key=lambda x: x["score"])}),
            flush=True,
        )

    baseline = np.asarray([record["rows"][0]["score"] for record in fold_records])
    summary = []
    for config_index, config in enumerate(CONFIGS, 1):
        values = np.asarray([record["rows"][config_index]["score"] for record in fold_records])
        delta = values - baseline
        summary.append(
            {
                "scale": config[0],
                "pdp_strength": config[1],
                "iterations": config[2],
                "scores": values.tolist(),
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
    summary.sort(key=lambda row: row["lcb"], reverse=True)
    payload = {"folds": fold_records, "summary": summary}
    (ROOT / "matched_phase6_safe_combo_channel_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:12]}), flush=True)


if __name__ == "__main__":
    run()
