from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .control_delta import MaskedProteomeAutoencoder, masked_mse, set_seed
from .losses import (MaskedLossConfig, fit_masked_loss_state, masked_loss,
                     masked_objective_numpy)


@dataclass
class AnchorBridgeConfig:
    hidden_dim: int = 256
    entity_projection_dim: int = 32
    dropout: float = 0.10
    modality_dropout: float = 0.20
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    sparsity_weight: float = 1e-4
    gate_weight: float = 1e-5
    patience: int = 8
    seed: int = 20260807


class GatedResidualBridge(nn.Module):
    def __init__(self, anchor_dim: int, context_dim: int, strain_dim: int,
                 chemical_dim: int, hidden_dim: int, projection_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.strain_projection = nn.Sequential(
            nn.Linear(strain_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU())
        self.chemical_projection = nn.Sequential(
            nn.Linear(chemical_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(anchor_dim + context_dim + 2 * projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.delta_head = nn.Linear(hidden_dim, anchor_dim)
        self.gate_head = nn.Linear(hidden_dim, anchor_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -3.0)

    def forward(self, anchor: torch.Tensor, context: torch.Tensor,
                strain: torch.Tensor, chemical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        merged = torch.cat([
            anchor, context, self.strain_projection(strain), self.chemical_projection(chemical)
        ], dim=1)
        hidden = self.trunk(merged)
        gate = torch.sigmoid(self.gate_head(hidden))
        return gate * self.delta_head(hidden), gate


def _drop_modality(x: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return x
    keep = (torch.rand((len(x), 1), device=x.device) >= probability).to(x.dtype)
    return x * keep


@torch.no_grad()
def encode_anchor(model: MaskedProteomeAutoencoder, anchor_prediction: np.ndarray,
                  protein_mean: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    result: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(anchor_prediction), batch_size):
        centered = np.asarray(anchor_prediction[start:start + batch_size], dtype=np.float32)
        centered = centered - protein_mean.reshape(1, -1)
        result.append(model.encode(torch.from_numpy(centered).to(device)).float().cpu().numpy())
    return np.concatenate(result).astype(np.float32)


@torch.no_grad()
def predict_bridge(model: GatedResidualBridge, decoder: MaskedProteomeAutoencoder,
                   anchor_prediction: np.ndarray, anchor_latent: np.ndarray,
                   context: np.ndarray, strain: np.ndarray, chemical: np.ndarray,
                   device: torch.device, batch_size: int,
                   zero_strain: bool = False, zero_chemical: bool = False) -> np.ndarray:
    model.eval()
    decoder.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(anchor_prediction), batch_size):
        stop = start + batch_size
        z = torch.from_numpy(anchor_latent[start:stop]).to(device)
        x_context = torch.from_numpy(context[start:stop]).to(device)
        x_strain = torch.from_numpy(strain[start:stop]).to(device)
        x_chemical = torch.from_numpy(chemical[start:stop]).to(device)
        if zero_strain:
            x_strain = torch.zeros_like(x_strain)
        if zero_chemical:
            x_chemical = torch.zeros_like(x_chemical)
        correction, _ = model(z, x_context, x_strain, x_chemical)
        # Decoder difference makes correction=0 an exact identity around the
        # original MLP anchor, regardless of autoencoder reconstruction error.
        protein_delta = decoder.decode(z + correction) - decoder.decode(z)
        anchor_block = np.array(anchor_prediction[start:stop], dtype=np.float32, copy=True)
        anchor = torch.from_numpy(anchor_block).to(device)
        output.append((anchor + protein_delta).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_anchor_bridge(anchor_prediction: np.ndarray, anchor_latent: np.ndarray,
                        context: np.ndarray, strain: np.ndarray, chemical: np.ndarray,
                        y_log2: np.ndarray, observation_mask: np.ndarray,
                        train_rows: np.ndarray, validation_rows: np.ndarray,
                        decoder: MaskedProteomeAutoencoder, protein_mean: np.ndarray,
                        config: AnchorBridgeConfig, output_dir: Path,
                        contract: dict[str, Any],
                        loss_config: MaskedLossConfig | None = None) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], GatedResidualBridge]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = loss_config or MaskedLossConfig()
    loss_state = fit_masked_loss_state(y_log2, observation_mask, train_rows, loss_config)
    protein_weights = torch.from_numpy(loss_state.protein_weights).to(device)
    decoder.to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    train_pos, valid_pos = np.flatnonzero(train_rows), np.flatnonzero(validation_rows)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(anchor_prediction[train_pos], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(anchor_latent[train_pos], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(context[train_pos], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(strain[train_pos], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(chemical[train_pos], dtype=np.float32)),
        torch.from_numpy(np.where(np.asarray(observation_mask)[train_pos],
                                  np.asarray(y_log2)[train_pos], 0.0).astype(np.float32)),
        torch.from_numpy(np.asarray(observation_mask)[train_pos].astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(config.seed))
    model = GatedResidualBridge(anchor_latent.shape[1], context.shape[1], strain.shape[1],
                                chemical.shape[1], config.hidden_dim,
                                config.entity_projection_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    valid_truth = np.asarray(y_log2)[valid_pos]
    valid_mask = np.asarray(observation_mask)[valid_pos]
    baseline_valid = np.asarray(anchor_prediction)[valid_pos]
    finite = valid_mask & np.isfinite(valid_truth)
    baseline_mse = masked_objective_numpy(baseline_valid, valid_truth, finite, loss_state)
    baseline_actual_mse = float(np.where(finite, (baseline_valid - valid_truth) ** 2, 0.0).sum() / finite.sum())
    best_loss, best_epoch, stale = baseline_mse, 0, 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    history: list[dict[str, float | int]] = [{
        "epoch": 0, "validation_objective": baseline_mse,
        "validation_log2_rmse": float(np.sqrt(baseline_actual_mse)),
        "selected": 1,
    }]

    for epoch in range(1, config.epochs + 1):
        model.train()
        for anchor, z, x_context, x_strain, x_chemical, target, mask in loader:
            anchor, z, x_context, x_strain, x_chemical, target, mask = [
                value.to(device, non_blocking=True) for value in
                (anchor, z, x_context, x_strain, x_chemical, target, mask)]
            x_strain = _drop_modality(x_strain, config.modality_dropout)
            x_chemical = _drop_modality(x_chemical, config.modality_dropout)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                correction, gate = model(z, x_context, x_strain, x_chemical)
                protein_delta = decoder.decode(z + correction) - decoder.decode(z)
                prediction = anchor + protein_delta
                loss = (masked_loss(prediction, target, mask, protein_weights, loss_config)
                        + config.sparsity_weight * correction.abs().mean()
                        + config.gate_weight * gate.mean())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        validation_prediction = predict_bridge(
            model, decoder, baseline_valid, anchor_latent[valid_pos], context[valid_pos],
            strain[valid_pos], chemical[valid_pos], device, config.batch_size)
        mse = masked_objective_numpy(validation_prediction, valid_truth, finite, loss_state)
        actual_mse = float(np.where(finite, (validation_prediction - valid_truth) ** 2, 0.0).sum() / finite.sum())
        selected = 0
        if mse < best_loss - 1e-6:
            best_loss, best_epoch, stale, selected = mse, epoch, 0, 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
        history.append({"epoch": epoch, "validation_objective": mse,
                        "validation_log2_rmse": float(np.sqrt(actual_mse)), "selected": selected})
        if stale >= config.patience:
            break

    model.load_state_dict(best_state)
    prediction = predict_bridge(model, decoder, baseline_valid, anchor_latent[valid_pos],
                                context[valid_pos], strain[valid_pos], chemical[valid_pos],
                                device, config.batch_size)
    final_actual_mse = float(np.where(finite, (prediction - valid_truth) ** 2, 0.0).sum() / finite.sum())
    output_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    torch.save({"state_dict": best_state, "config": asdict(config), "contract": contract,
                "best_epoch": best_epoch, "best_validation_masked_mse": best_loss,
                "baseline_validation_masked_mse": baseline_mse,
                "loss": loss_state.summary()}, output_dir / "anchor_bridge.pt")
    info = {"best_epoch": best_epoch, "best_validation_objective": best_loss,
            "best_validation_log2_rmse": float(np.sqrt(final_actual_mse)),
            "baseline_validation_log2_rmse": float(np.sqrt(baseline_actual_mse)),
            "improved_over_anchor": bool(best_epoch > 0), "epochs_ran": len(history) - 1,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "device": str(device), "config": asdict(config), "loss": loss_state.summary()}
    return prediction, history_frame, info, model
