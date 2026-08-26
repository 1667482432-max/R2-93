from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

import r2_pipeline as rp

os.environ.setdefault("R2_BANDS", "24")
from matched_phase5_tree_band_descriptor import BANDS, build_cache, normalize_last
from matched_phase6_pas_mlp_descriptor import ResidualBlock, fourier_features


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")
EPOCHS = 650
HIDDEN = 512
COMPONENTS = 192


class RichSpatialMLP(torch.nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.input = torch.nn.Linear(inputs, HIDDEN)
        self.blocks = torch.nn.ModuleList([ResidualBlock(HIDDEN) for _ in range(5)])
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
    scale = np.maximum(features[train].std(0), 1e-5)
    x_train = torch.as_tensor((features[train] - center) / scale, device=DEVICE)
    x_val = torch.as_tensor((features[val] - center) / scale, device=DEVICE)
    coefficient_scale = max(float(np.std(coefficients)), 1e-6)
    y_train = torch.as_tensor(coefficients / coefficient_scale, device=DEVICE)
    model = RichSpatialMLP(x_train.shape[1], y_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=3e-5
    )
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed + 1000)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(x_train), generator=generator, device=DEVICE)
        for start in range(0, len(order), 512):
            index = order[start : start + 512]
            prediction = model(x_train[index])
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, y_train[index], beta=0.15
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        if epoch in (0, 199, 399, EPOCHS - 1):
            print(
                json.dumps({"stage": "epoch", "seed": seed, "epoch": epoch + 1, "loss": float(loss)}),
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        return model(x_val).cpu().numpy() * coefficient_scale


def run() -> None:
    pos, _, energy = rp.load_data()
    target = build_cache()
    test_pos = np.load(ROOT / "Round2_Test_Pos.npy")
    all_pos = np.vstack((pos, test_pos))
    base_features = fourier_features(all_pos, np.load(ROOT / "los_map_features.npy"))
    rich = np.load(ROOT / "rich_map_features.npy").astype(np.float32)
    features = np.column_stack((base_features, rich)).astype(np.float32)
    valid = np.flatnonzero(energy > 0)
    for fold in range(5):
        val = np.load(ROOT / f"matched_rect_val_{fold}.npy")
        train = valid[~np.isin(valid, val)]
        shaped = np.asarray(target[train]).transpose(0, 2, 3, 1)
        transformed = np.sqrt(np.maximum(shaped, 0)).reshape(len(train), -1)
        pca = PCA(
            n_components=COMPONENTS,
            svd_solver="randomized",
            n_oversamples=20,
            iterated_power=4,
            random_state=38100 + fold,
        )
        coefficients = pca.fit_transform(transformed)
        predicted_coefficients = fit_predict(
            features, coefficients, train, val, 38200 + fold
        )
        prediction = pca.inverse_transform(predicted_coefficients)
        prediction = np.maximum(
            prediction.reshape(len(val), 4, BANDS, 256), 0
        ) ** 2
        prediction = normalize_last(prediction).transpose(0, 3, 1, 2).astype(np.float32)
        np.save(ROOT / f"matched_rich_mlp_pas_band{BANDS}_fold{fold}.npy", prediction)
        print(json.dumps({"stage": "fold", "fold": fold}), flush=True)


if __name__ == "__main__":
    run()
