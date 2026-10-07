from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import MATCH_KEYS
from .losses import (MaskedLossConfig, fit_masked_loss_state, masked_loss,
                     masked_objective_numpy)


@dataclass
class ControlDeltaConfig:
    latent_dim: int = 128
    autoencoder_hidden_dim: int = 512
    condition_hidden_dim: int = 256
    batch_size: int = 128
    autoencoder_epochs: int = 35
    control_epochs: int = 60
    delta_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    n_control_folds: int = 5
    seed: int = 20260806


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MaskedProteomeAutoencoder(nn.Module):
    def __init__(self, protein_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, protein_dim),
        )

    def encode(self, centered_filled: torch.Tensor) -> torch.Tensor:
        return self.encoder(centered_filled)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, centered_filled: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(centered_filled))


class LatentRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def context_fold_ids(metadata: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Assign exact experimental contexts to stable folds without Python hash randomization."""
    result = np.empty(len(metadata), dtype=np.int64)
    for i, row in enumerate(metadata.loc[:, MATCH_KEYS].itertuples(index=False, name=None)):
        token = "\x1f".join(map(str, row)).encode("utf-8")
        result[i] = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % n_folds
    return result


@torch.no_grad()
def predict_batches(model: nn.Module, features: np.ndarray, device: torch.device,
                    batch_size: int) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start + batch_size]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            value = model(batch)
        output.append(value.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def encode_proteomes(model: MaskedProteomeAutoencoder, values: np.ndarray,
                     protein_mean: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        block = np.asarray(values[start:start + batch_size])
        centered = np.where(np.isfinite(block), block - protein_mean.reshape(1, -1), 0.0).astype(np.float32)
        result.append(model.encode(torch.from_numpy(centered).to(device)).float().cpu().numpy())
    return np.concatenate(result).astype(np.float32)


@torch.no_grad()
def decode_proteomes(model: MaskedProteomeAutoencoder, latent: np.ndarray,
                     protein_mean: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    for start in range(0, len(latent), batch_size):
        block = torch.from_numpy(latent[start:start + batch_size]).to(device)
        result.append(model.decode(block).float().cpu().numpy())
    return (np.concatenate(result) + protein_mean.reshape(1, -1)).astype(np.float32)


def train_autoencoder(y_log2: np.ndarray, mask: np.ndarray, train_rows: np.ndarray,
                      validation_rows: np.ndarray, protein_mean: np.ndarray,
                      config: ControlDeltaConfig, output_dir: Path,
                      loss_config: MaskedLossConfig | None = None) -> tuple[MaskedProteomeAutoencoder, dict[str, Any]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = loss_config or MaskedLossConfig()
    loss_state = fit_masked_loss_state(y_log2, mask, train_rows, loss_config)
    protein_weights = torch.from_numpy(loss_state.protein_weights).to(device)
    train_pos = np.flatnonzero(train_rows)
    valid_pos = np.flatnonzero(validation_rows)
    train_y = np.asarray(y_log2)[train_pos]
    train_mask = np.asarray(mask)[train_pos]
    centered = np.where(train_mask, train_y - protein_mean.reshape(1, -1), 0.0).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(centered), torch.from_numpy(train_mask.astype(np.float32))),
                        batch_size=config.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(config.seed))
    model = MaskedProteomeAutoencoder(y_log2.shape[1], config.autoencoder_hidden_dim, config.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history: list[dict[str, float | int]] = []
    valid_y = np.asarray(y_log2)[valid_pos]
    valid_mask = np.asarray(mask)[valid_pos]
    valid_centered = np.where(valid_mask, valid_y - protein_mean.reshape(1, -1), 0.0).astype(np.float32)
    for epoch in range(1, config.autoencoder_epochs + 1):
        model.train()
        for batch_x, batch_mask in loader:
            batch_x, batch_mask = batch_x.to(device), batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = masked_loss(model(batch_x), batch_x, batch_mask,
                                   protein_weights, loss_config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        validation_blocks: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(valid_pos), config.batch_size):
                x = torch.from_numpy(valid_centered[start:start + config.batch_size]).to(device)
                m = torch.from_numpy(valid_mask[start:start + config.batch_size].astype(np.float32)).to(device)
                prediction = model(x)
                validation_blocks.append(prediction.float().cpu().numpy())
        validation_prediction = np.concatenate(validation_blocks)
        objective = masked_objective_numpy(validation_prediction, valid_centered, valid_mask, loss_state)
        finite = valid_mask & np.isfinite(valid_centered)
        mse = float(np.where(finite, (validation_prediction - valid_centered) ** 2, 0.0).sum() / finite.sum())
        history.append({"epoch": epoch, "validation_objective": objective,
                        "validation_reconstruction_log2_rmse": float(np.sqrt(mse))})
        if objective < best_loss - 1e-6:
            best_loss, best_epoch, stale = objective, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_rmse = float(np.sqrt(mse))
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Autoencoder did not produce a checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "autoencoder_history.csv", index=False)
    torch.save({"state_dict": best_state, "config": asdict(config), "protein_mean": protein_mean,
                "protein_dim": int(y_log2.shape[1]), "best_epoch": best_epoch,
                "best_validation_objective": best_loss, "loss": loss_state.summary()}, output_dir / "proteome_autoencoder.pt")
    return model, {"best_epoch": best_epoch, "best_validation_objective": best_loss,
                   "validation_reconstruction_log2_rmse": best_rmse, "loss": loss_state.summary(),
                   "parameter_count": int(sum(p.numel() for p in model.parameters())), "device": str(device)}


def train_latent_regressor(features: np.ndarray, targets: np.ndarray, train_mask: np.ndarray,
                           valid_mask: np.ndarray, config: ControlDeltaConfig, epochs: int,
                           seed: int) -> tuple[LatentRegressor, dict[str, Any]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pos, valid_pos = np.flatnonzero(train_mask), np.flatnonzero(valid_mask)
    model = LatentRegressor(features.shape[1], targets.shape[1], config.condition_hidden_dim).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(features[train_pos], dtype=np.float32)),
                                      torch.from_numpy(np.ascontiguousarray(targets[train_pos], dtype=np.float32))),
                        batch_size=config.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - target).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        prediction = predict_batches(model, np.ascontiguousarray(features[valid_pos], dtype=np.float32),
                                     device, config.batch_size)
        mse = float(np.mean((prediction - targets[valid_pos]) ** 2))
        history.append({"epoch": epoch, "validation_latent_mse": mse})
        if mse < best_loss - 1e-7:
            best_loss, best_epoch, stale = mse, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Latent regressor did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_latent_mse": best_loss,
                   "history": history, "parameter_count": int(sum(p.numel() for p in model.parameters()))}


def fit_pca_embedding(entity_features: np.ndarray, output_dim: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Label-free entity PCA; all rows are external covariates, never proteome targets."""
    x = np.asarray(entity_features, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    width = min(output_dim, max(1, len(x) - 1))
    scores = u[:, :width] * s[:width]
    scale = scores.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (scores / scale).astype(np.float32), {
        "mean": mean.astype(np.float32), "components": vt[:width].astype(np.float32),
        "score_scale": scale.astype(np.float32),
    }
