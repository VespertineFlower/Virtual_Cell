from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .constants import CONTROL_NAMES, EXCLUDED_TREATMENTS
from .control_delta import (MaskedProteomeAutoencoder, decode_proteomes,
                            encode_proteomes)
from .mlp import set_reproducible_seed
from .physics_residual import make_time_triples


@dataclass(frozen=True)
class LatentBridgeConfig:
    hidden_dim: int = 512
    dropout: float = 0.1
    batch_size: int = 128
    auxiliary_batch_size: int = 4
    pretrain_epochs: int = 30
    finetune_epochs: int = 8
    patience: int = 7
    learning_rate: float = 5e-4
    finetune_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    interpolation_noise: float = 0.10
    finetune_steps: int = 4
    control_weight: float = 0.05
    curvature_weight: float = 0.05
    seed: int = 20260804


class ConditionalLatentBridge(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 2 * latent_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, noisy_residual: torch.Tensor, anchor_latent: torch.Tensor,
                features: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.network(torch.cat([noisy_residual, anchor_latent, features, time], dim=1))


def sample_latent_bridge(model: ConditionalLatentBridge, anchor_latent: torch.Tensor,
                         features: torch.Tensor, steps: int,
                         stochastic: bool = False, noise_scale: float = 0.05,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    residual = torch.zeros_like(anchor_latent)
    schedule = torch.linspace(1.0, 0.0, steps + 1, device=anchor_latent.device)
    for index in range(steps):
        current = schedule[index].expand(len(anchor_latent))
        following = schedule[index + 1]
        endpoint = model(residual, anchor_latent, features, current)
        residual = (1.0 - following) * endpoint
        if stochastic and float(following) > 0:
            noise = torch.randn(residual.shape, device=residual.device,
                                dtype=residual.dtype, generator=generator)
            residual = residual + noise_scale * torch.sqrt(
                following * (1.0 - following)) * noise
    return residual


@torch.no_grad()
def predict_bridge_correction(model: ConditionalLatentBridge,
                              autoencoder: MaskedProteomeAutoencoder,
                              anchor: np.ndarray, features: np.ndarray,
                              protein_mean: np.ndarray, device: torch.device,
                              batch_size: int, steps: int,
                              stochastic_samples: int = 0, seed: int = 20260804,
                              ) -> np.ndarray:
    model.eval(); autoencoder.eval()
    anchor_latent = encode_proteomes(autoencoder, anchor, protein_mean, device, batch_size)
    corrections = []
    for start in range(0, len(anchor), batch_size):
        z_anchor = torch.from_numpy(anchor_latent[start:start + batch_size]).to(device)
        batch_features = torch.from_numpy(
            np.asarray(features[start:start + batch_size], dtype=np.float32)).to(device)
        if stochastic_samples > 0:
            samples = []
            for sample_index in range(stochastic_samples):
                generator = torch.Generator(device=device).manual_seed(seed + sample_index)
                samples.append(sample_latent_bridge(
                    model, z_anchor, batch_features, steps, stochastic=True,
                    generator=generator))
            residual = torch.stack(samples).mean(dim=0)
        else:
            residual = sample_latent_bridge(model, z_anchor, batch_features, steps)
        decoded_base = autoencoder.decode(z_anchor)
        decoded_shifted = autoencoder.decode(z_anchor + residual)
        corrections.append((decoded_shifted - decoded_base).float().cpu().numpy())
    return np.concatenate(corrections).astype(np.float32)


def train_latent_diffusion_bridge(
    features: np.ndarray,
    metadata: pd.DataFrame,
    y_log2: np.ndarray,
    observation_mask: np.ndarray,
    full_anchor: np.ndarray,
    train_rows: np.ndarray,
    selection_rows: np.ndarray,
    autoencoder: MaskedProteomeAutoencoder,
    protein_mean: np.ndarray,
    config: LatentBridgeConfig,
    output_dir: Path,
) -> tuple[ConditionalLatentBridge, pd.DataFrame, dict[str, Any]]:
    set_reproducible_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device).eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    train_treatments = train_rows & ~metadata["perturbation_no_concentration"].isin(
        EXCLUDED_TREATMENTS).to_numpy()
    train_positions = np.flatnonzero(train_treatments)
    truth = np.asarray(y_log2)
    mask = np.asarray(observation_mask).astype(bool)
    z_truth = encode_proteomes(autoencoder, truth[train_positions], protein_mean,
                               device, config.batch_size)
    z_anchor_train = encode_proteomes(autoencoder, full_anchor[train_positions],
                                      protein_mean, device, config.batch_size)
    latent_target = z_truth - z_anchor_train
    protein_residual = np.where(
        mask[train_positions] & np.isfinite(full_anchor[train_positions]),
        truth[train_positions] - full_anchor[train_positions], 0.0,
    ).astype(np.float32)
    protein_mask = (mask[train_positions] &
                    np.isfinite(full_anchor[train_positions])).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(features[train_positions], dtype=np.float32)),
        torch.from_numpy(z_anchor_train), torch.from_numpy(latent_target),
        torch.from_numpy(protein_residual), torch.from_numpy(protein_mask),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(config.seed))
    latent_dim = z_anchor_train.shape[1]
    model = ConditionalLatentBridge(features.shape[1], latent_dim,
                                    config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    selection_positions = np.flatnonzero(selection_rows)
    selection_anchor = full_anchor[selection_positions]
    selection_features = features[selection_positions]
    selection_truth = truth[selection_positions]
    selection_mask = mask[selection_positions] & np.isfinite(selection_truth)

    def evaluate() -> float:
        correction = predict_bridge_correction(
            model, autoencoder, selection_anchor, selection_features,
            protein_mean, device, config.batch_size, config.finetune_steps)
        error = selection_anchor + correction - selection_truth
        mse = float(np.where(selection_mask, error * error, 0.0).sum()
                    / max(selection_mask.sum(), 1))
        return float(np.sqrt(mse))

    baseline_error = selection_anchor - selection_truth
    baseline_rmse = float(np.sqrt(np.where(
        selection_mask, baseline_error * baseline_error, 0.0).sum()
        / max(selection_mask.sum(), 1)))
    best_rmse = baseline_rmse
    best_epoch = 0
    best_stage = "anchor"
    best_state = copy.deepcopy({key: value.detach().cpu()
                                for key, value in model.state_dict().items()})
    history: list[dict[str, Any]] = [{
        "stage": "anchor", "epoch": 0, "train_loss": np.nan,
        "selection_log2_rmse": baseline_rmse,
    }]
    stale = 0
    rng = torch.Generator(device=device).manual_seed(config.seed)
    for epoch in range(1, config.pretrain_epochs + 1):
        model.train(); values = []
        for batch_features, batch_anchor, batch_target, _, _ in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_anchor = batch_anchor.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            time = torch.rand(len(batch_target), 1, device=device, generator=rng) * 0.98 + 0.02
            noise = torch.randn(batch_target.shape, device=device, generator=rng)
            noisy = ((1.0 - time) * batch_target + config.interpolation_noise *
                     torch.sqrt(time * (1.0 - time)) * noise)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                predicted = model(noisy, batch_anchor, batch_features, time)
                loss = F.mse_loss(predicted, batch_target)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); values.append(float(loss.item()))
        rmse = evaluate()
        history.append({"stage": "pretrain", "epoch": epoch,
                        "train_loss": float(np.mean(values)),
                        "selection_log2_rmse": rmse})
        if rmse < best_rmse - 1e-6:
            best_rmse, best_epoch, best_stage = rmse, epoch, "pretrain"
            best_state = copy.deepcopy({key: value.detach().cpu()
                                        for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_learning_rate,
                                  weight_decay=config.weight_decay)
    triples, weights = make_time_triples(metadata, train_rows)
    z_anchor_all = encode_proteomes(autoencoder, full_anchor, protein_mean,
                                    device, config.batch_size)
    control_positions = np.flatnonzero(
        train_rows & metadata["perturbation_no_concentration"].isin(CONTROL_NAMES).to_numpy())
    control_features = torch.from_numpy(
        np.asarray(features[control_positions], dtype=np.float32)).to(device)
    control_anchor = torch.from_numpy(z_anchor_all[control_positions]).to(device)
    triple_features = torch.from_numpy(
        np.asarray(features[triples], dtype=np.float32)).to(device) if len(triples) else None
    triple_anchor = torch.from_numpy(z_anchor_all[triples]).to(device) if len(triples) else None
    triple_weights = torch.from_numpy(weights).to(device) if len(triples) else None
    np_rng = np.random.default_rng(config.seed + 101)
    stale = 0
    for epoch in range(1, config.finetune_epochs + 1):
        model.train(); values = []
        for batch_features, batch_anchor, batch_target, batch_protein, batch_mask in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_anchor = batch_anchor.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_protein = batch_protein.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                residual = sample_latent_bridge(
                    model, batch_anchor, batch_features, config.finetune_steps)
                protein_correction = (autoencoder.decode(batch_anchor + residual) -
                                      autoencoder.decode(batch_anchor))
                match = F.huber_loss(protein_correction, batch_protein,
                                     reduction="none", delta=1.0)
                match = (match * batch_mask).sum() / batch_mask.sum().clamp_min(1.0)
                latent_match = F.mse_loss(residual, batch_target)
                control_loss = residual.sum() * 0.0
                if config.control_weight > 0 and len(control_positions):
                    selected = np_rng.integers(len(control_positions),
                                               size=config.auxiliary_batch_size)
                    control_residual = sample_latent_bridge(
                        model, control_anchor[selected], control_features[selected],
                        config.finetune_steps)
                    control_loss = F.huber_loss(control_residual,
                                                torch.zeros_like(control_residual))
                curvature_loss = residual.sum() * 0.0
                if config.curvature_weight > 0 and triple_features is not None:
                    selected = np_rng.integers(len(triples), size=config.auxiliary_batch_size)
                    z = triple_anchor[selected].reshape(-1, latent_dim)
                    x = triple_features[selected].reshape(-1, features.shape[1])
                    triple_residual = sample_latent_bridge(
                        model, z, x, config.finetune_steps).reshape(len(selected), 3, latent_dim)
                    w = triple_weights[selected]
                    interpolated = (w[:, :1] * triple_residual[:, 0] +
                                    w[:, 1:] * triple_residual[:, 2])
                    curvature_loss = F.huber_loss(
                        triple_residual[:, 1] - interpolated,
                        torch.zeros_like(triple_residual[:, 1]))
                loss = (match + 0.1 * latent_match + config.control_weight * control_loss +
                        config.curvature_weight * curvature_loss)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); values.append(float(loss.item()))
        rmse = evaluate()
        history.append({"stage": "finetune", "epoch": epoch,
                        "train_loss": float(np.mean(values)),
                        "selection_log2_rmse": rmse})
        if rmse < best_rmse - 1e-6:
            best_rmse, best_epoch, best_stage = rmse, epoch, "finetune"
            best_state = copy.deepcopy({key: value.detach().cpu()
                                        for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state); model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = {
        "state_dict": best_state, "config": asdict(config),
        "feature_dim": features.shape[1], "latent_dim": latent_dim,
        "best_stage": best_stage, "best_epoch": best_epoch,
        "best_selection_log2_rmse": best_rmse,
        "baseline_selection_log2_rmse": baseline_rmse,
        "time_triples": len(triples), "control_samples": len(control_positions),
    }
    torch.save(checkpoint, output_dir / "latent_diffusion_bridge.pt")
    info = {**checkpoint, "state_dict": None,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "device": str(device)}
    return model, history_frame, info


def load_autoencoder(checkpoint_path: Path, device: torch.device
                     ) -> tuple[MaskedProteomeAutoencoder, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MaskedProteomeAutoencoder(
        int(checkpoint["protein_dim"]), int(config["autoencoder_hidden_dim"]),
        int(config["latent_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), np.asarray(checkpoint["protein_mean"], dtype=np.float32)
