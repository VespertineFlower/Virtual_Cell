from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import (MaskedLossConfig, fit_masked_loss_state, masked_loss,
                     masked_objective_numpy, masked_rowwise_pearson_loss)


@dataclass
class MLPConfig:
    hidden_dim: int = 256
    dropout: float = 0.10
    batch_size: int = 128
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 20260804


class ConditionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        # The model predicts a residual around Protein Mean. Zero initialization
        # makes the first prediction exactly the statistical baseline.
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (prediction - target).square() * mask
    return squared.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def predict_residuals(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(batch)
        outputs.append(prediction.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_condition_mlp(
    features: np.ndarray,
    y_log2: np.ndarray,
    observation_mask: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    protein_mean_vector: np.ndarray,
    config: MLPConfig,
    output_dir: Path,
    encoder_info: dict[str, Any],
    protein_names: list[str],
    loss_config: MaskedLossConfig | None = None,
    control_reference: np.ndarray | None = None,
    fc_correlation_weight: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    set_reproducible_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = loss_config or MaskedLossConfig()
    loss_state = fit_masked_loss_state(y_log2, observation_mask, train_rows, loss_config)
    protein_weights = torch.from_numpy(loss_state.protein_weights).to(device)

    train_positions = np.flatnonzero(train_rows)
    validation_positions = np.flatnonzero(validation_rows)
    train_features = np.ascontiguousarray(features[train_positions], dtype=np.float32)
    validation_features = np.ascontiguousarray(features[validation_positions], dtype=np.float32)

    train_truth = np.asarray(y_log2)[train_positions]
    train_mask = np.asarray(observation_mask)[train_positions]
    train_residual = np.where(
        train_mask,
        train_truth - protein_mean_vector.reshape(1, -1),
        0.0,
    ).astype(np.float32)
    if fc_correlation_weight < 0:
        raise ValueError("fc_correlation_weight must be non-negative")
    if control_reference is None:
        train_control = np.zeros_like(train_residual)
        train_fc_mask = np.zeros_like(train_mask, dtype=np.float32)
    else:
        reference = np.asarray(control_reference)[train_positions]
        reference_finite = np.isfinite(reference)
        train_control = np.where(reference_finite, reference, 0.0).astype(np.float32)
        train_fc_mask = (train_mask & reference_finite).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_residual),
        torch.from_numpy(train_mask.astype(np.float32)),
        torch.from_numpy(train_control),
        torch.from_numpy(train_fc_mask),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    model = ConditionMLP(
        input_dim=features.shape[1],
        output_dim=y_log2.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    validation_truth = np.asarray(y_log2)[validation_positions]
    validation_mask = np.asarray(observation_mask)[validation_positions]
    best_loss = float("inf")
    best_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_squared_sum = 0.0
        train_observed = 0.0
        for batch_features, batch_target, batch_mask, batch_control, batch_fc_mask in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            batch_control = batch_control.to(device, non_blocking=True)
            batch_fc_mask = batch_fc_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(batch_features)
                loss = masked_loss(prediction, batch_target, batch_mask, protein_weights, loss_config)
                if fc_correlation_weight > 0:
                    mean_tensor = torch.from_numpy(protein_mean_vector).to(device)
                    delta_prediction = prediction + mean_tensor - batch_control
                    delta_target = batch_target + mean_tensor - batch_control
                    loss = loss + fc_correlation_weight * masked_rowwise_pearson_loss(
                        delta_prediction, delta_target, batch_fc_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            observed = float(batch_mask.sum().item())
            train_squared_sum += float(loss.item()) * observed
            train_observed += observed

        residual_prediction = predict_residuals(
            model, validation_features, device, config.batch_size
        )
        absolute_prediction = residual_prediction + protein_mean_vector.reshape(1, -1)
        validation_loss = masked_objective_numpy(
            absolute_prediction, validation_truth, validation_mask, loss_state)
        valid = validation_mask & np.isfinite(validation_truth)
        validation_mse = float(np.where(valid, (absolute_prediction - validation_truth) ** 2, 0.0).sum()
                               / valid.sum())
        train_loss = train_squared_sum / max(train_observed, 1.0)
        history.append(
            {
                "epoch": epoch,
                "train_masked_mse": train_loss,
                "validation_objective": validation_loss,
                "validation_log2_rmse": float(np.sqrt(validation_mse)),
            }
        )

        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_rmse = float(np.sqrt(validation_mse))
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    all_residuals = predict_residuals(model, features, device, config.batch_size)
    all_predictions = all_residuals + protein_mean_vector.reshape(1, -1)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "input_dim": int(features.shape[1]),
        "output_dim": int(y_log2.shape[1]),
        "best_epoch": best_epoch,
        "best_validation_masked_mse": best_loss,
        "best_validation_objective": best_loss,
        "best_validation_log2_rmse": best_rmse,
        "protein_mean": protein_mean_vector,
        "protein_names": protein_names,
        "encoder": encoder_info,
        "loss": loss_state.summary(),
        "fc_correlation_weight": fc_correlation_weight,
    }
    torch.save(checkpoint, output_dir / "condition_mlp.pt")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    run_info = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "best_validation_masked_mse": best_loss,
        "best_validation_log2_rmse": best_rmse,
        "best_validation_objective": best_loss,
        "epochs_ran": len(history),
        "config": asdict(config),
        "loss": loss_state.summary(),
        "fc_correlation_weight": fc_correlation_weight,
    }
    return all_predictions.astype(np.float32), history_frame, run_info


def load_checkpoint_predictions(
    checkpoint_path: Path,
    features: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = ConditionMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    residual = predict_residuals(model, features, device, batch_size)
    mean = np.asarray(checkpoint["protein_mean"], dtype=np.float32).reshape(1, -1)
    return (residual + mean).astype(np.float32), checkpoint
