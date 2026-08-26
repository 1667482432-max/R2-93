from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

import r2_pipeline as rp
os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import BANDS, base_descriptor, build_cache, normalize_last
from matched_phase5_tree_descriptor import model_features


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
EPOCHS = 900
HIDDEN = 384


def fourier_features(pos: np.ndarray, map_features: np.ndarray) -> np.ndarray:
    base = model_features(pos, map_features)
    xy = pos[:, :2]
    side = xy[:, 1] > 0
    bs = np.where(side[:, None], np.array([52.0, 35.0]), np.array([-18.413, -65.881]))
    relative = xy - bs
    radius = np.linalg.norm(relative, axis=1)
    blocks = [base]
    for wavelength in (10.0, 20.0, 40.0, 80.0, 160.0, 320.0):
        phase_x = 2 * np.pi * xy[:, 0] / wavelength
        phase_y = 2 * np.pi * xy[:, 1] / wavelength
        phase_r = 2 * np.pi * radius / wavelength
        blocks.append(
            np.column_stack(
                (
                    np.sin(phase_x), np.cos(phase_x),
                    np.sin(phase_y), np.cos(phase_y),
                    np.sin(phase_r), np.cos(phase_r),
                )
            )
        )
    return np.column_stack(blocks).astype(np.float32)


class ResidualBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.linear1 = torch.nn.Linear(width, width)
        self.linear2 = torch.nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(self.norm(value))
        hidden = torch.nn.functional.silu(hidden)
        hidden = self.linear2(hidden)
        return value + 0.25 * hidden


class SpatialMLP(torch.nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.input = torch.nn.Linear(inputs, HIDDEN)
        self.blocks = torch.nn.ModuleList([ResidualBlock(HIDDEN) for _ in range(4)])
        self.output_norm = torch.nn.LayerNorm(HIDDEN)
        self.output = torch.nn.Linear(HIDDEN, outputs)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.silu(self.input(value))
        for block in self.blocks:
            value = block(value)
        return self.output(torch.nn.functional.silu(self.output_norm(value)))


def fit_predict(
    features: np.ndarray,
    coefficients: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    center = features[train].mean(0)
    scale = features[train].std(0)
    scale = np.maximum(scale, 1e-5)
    x_train = torch.as_tensor((features[train] - center) / scale, device=DEVICE)
    x_val = torch.as_tensor((features[val] - center) / scale, device=DEVICE)
    coefficient_scale = max(float(np.std(coefficients)), 1e-6)
    y_train = torch.as_tensor(coefficients / coefficient_scale, device=DEVICE)
    model = SpatialMLP(x_train.shape[1], y_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=5e-5
    )
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed + 1000)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(x_train), generator=generator, device=DEVICE)
        for start in range(0, len(order), 512):
            index = order[start:start + 512]
            prediction = model(x_train[index])
            loss = torch.nn.functional.mse_loss(prediction, y_train[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        if epoch in (0, 299, 599, 1199, EPOCHS - 1):
            print(json.dumps({"stage": "epoch", "seed": seed, "epoch": epoch + 1, "loss": float(loss)}), flush=True)
    model.eval()
    with torch.no_grad():
        result = model(x_val).cpu().numpy() * coefficient_scale
    return result


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    all_pos = np.vstack((pos, test_pos))
    features = fourier_features(all_pos, np.load(ROOT / "los_map_features.npy"))
    valid = np.flatnonzero(energy > 0)
    test_groups = rp.official_island_labels(test_pos)
    counts = {int(group): int(np.sum(test_groups == group)) for group in np.unique(test_groups)}
    fold_data = []
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        labels = np.load(ROOT / f"matched_rect_groups_{fold}.npy").astype(np.int64)
        train = valid[~np.isin(valid, val)]
        shaped = np.asarray(target[train]).transpose(0, 2, 3, 1)
        transformed = np.sqrt(np.maximum(shaped, 0)).reshape(len(train), -1)
        pca = PCA(n_components=160, svd_solver="randomized", random_state=27100 + fold)
        coefficients = pca.fit_transform(transformed)
        predicted_coefficients = fit_predict(
            features, coefficients, train, val, 27200 + fold
        )
        prediction = pca.inverse_transform(predicted_coefficients)
        prediction = np.maximum(prediction.reshape(len(val), 4, BANDS, 256), 0) ** 2
        prediction = normalize_last(prediction).transpose(0, 3, 1, 2).astype(np.float32)
        np.save(ROOT / f"matched_mlp_pas_band{BANDS}_fold{fold}.npy", prediction)
        base_path = ROOT / f"matched_phase5_pas_band{BANDS}_fold{fold}.npy"
        if base_path.exists():
            base = np.load(base_path, mmap_mode="r")
        else:
            base = base_descriptor(
                np.load(ROOT / f"matched_phase5_full_fold{fold}.npy", mmap_mode="r")
            )
            np.save(base_path, base)
        tree = np.load(ROOT / f"matched_tree_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
        rbf = np.load(ROOT / f"matched_rbf_pas_band{BANDS}_fold{fold}.npy", mmap_mode="r")
        truth = np.asarray(target[val])
        weights = np.asarray([counts[int(group)] / np.sum(labels == group) for group in labels])
        fold_data.append((base, tree, rbf, prediction, truth, weights))
        print(json.dumps({"stage": "fold", "fold": fold}), flush=True)

    baseline = []
    for base, _, _, _, truth, weights in fold_data:
        cosine = np.sum(base * truth, axis=3) / np.maximum(
            np.linalg.norm(base, axis=3) * np.linalg.norm(truth, axis=3), 1e-30
        )
        baseline.append(float(np.sum(weights * cosine.mean((1, 2))) / weights.sum()))
    baseline = np.asarray(baseline)
    configs = []
    for tree_weight in (0.10, 0.20, 0.25, 0.30, 0.35):
        for rbf_weight in (0.00, 0.05, 0.10, 0.15, 0.20):
            for mlp_weight in (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
                if 0 < tree_weight + rbf_weight + mlp_weight <= 0.8:
                    configs.append((tree_weight, rbf_weight, mlp_weight))
    summary = []
    for tree_weight, rbf_weight, mlp_weight in configs:
        values = []
        for base, tree, rbf, mlp, truth, weights in fold_data:
            prediction = normalize_last(
                (1.0 - tree_weight - rbf_weight - mlp_weight) * base
                + tree_weight * tree
                + rbf_weight * rbf
                + mlp_weight * mlp
            )
            cosine = np.sum(prediction * truth, axis=3) / np.maximum(
                np.linalg.norm(prediction, axis=3) * np.linalg.norm(truth, axis=3), 1e-30
            )
            values.append(float(np.sum(weights * cosine.mean((1, 2))) / weights.sum()))
        delta = np.asarray(values) - baseline
        summary.append(
            {
                "tree_weight": tree_weight,
                "rbf_weight": rbf_weight,
                "mlp_weight": mlp_weight,
                "deltas": delta.tolist(),
                "mean_delta": float(delta.mean()),
                "min_delta": float(delta.min()),
                "lcb": float(delta.mean() - 0.75 * delta.std()),
            }
        )
    summary.sort(key=lambda item: item["lcb"], reverse=True)
    (ROOT / f"matched_phase6_pas_mlp_band{BANDS}_descriptor.json").write_text(
        json.dumps({"summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"top": summary[:30]}), flush=True)


if __name__ == "__main__":
    run()
