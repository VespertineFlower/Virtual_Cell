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
from .mlp import predict_residuals, set_reproducible_seed


TIME_GROUP_KEYS = (
    "data_source", "instrument", "Strains", "perturbation_no_concentration",
    "Medium", "Temperature", "pert_time_unit",
)


@dataclass(frozen=True)
class PhysicsResidualConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 128
    auxiliary_batch_size: int = 8
    epochs: int = 30
    patience: int = 7
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    control_weight: float = 0.0
    curvature_weight: float = 0.0
    seed: int = 20260804


class PhysicsResidualCorrector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def make_time_triples(metadata: pd.DataFrame, train_rows: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return consecutive-time row triples and linear interpolation weights."""
    eligible = train_rows & ~metadata["perturbation_no_concentration"].isin(
        EXCLUDED_TREATMENTS).to_numpy()
    groups: dict[tuple[object, ...], dict[float, int]] = {}
    for position in np.flatnonzero(eligible):
        row = metadata.iloc[position]
        key = tuple(row[field] for field in TIME_GROUP_KEYS)
        time = float(row["pert_time"])
        groups.setdefault(key, {}).setdefault(time, int(position))
    triples, weights = [], []
    for time_map in groups.values():
        times = sorted(time_map)
        for left, middle, right in zip(times, times[1:], times[2:]):
            span = right - left
            if span <= 0:
                continue
            triples.append([time_map[left], time_map[middle], time_map[right]])
            weights.append([(right - middle) / span, (middle - left) / span])
    if not triples:
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 2), dtype=np.float32)
    return np.asarray(triples, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def _masked_huber(prediction: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    loss = F.huber_loss(prediction, target, reduction="none", delta=1.0)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def train_physics_residual(
    features: np.ndarray,
    metadata: pd.DataFrame,
    y_log2: np.ndarray,
    observation_mask: np.ndarray,
    oof_anchor: np.ndarray,
    validation_anchor: np.ndarray,
    train_rows: np.ndarray,
    selection_rows: np.ndarray,
    config: PhysicsResidualConfig,
    output_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    set_reproducible_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_treatments = train_rows & ~metadata["perturbation_no_concentration"].isin(
        EXCLUDED_TREATMENTS).to_numpy()
    train_positions = np.flatnonzero(train_treatments)
    truth = np.asarray(y_log2)
    mask = np.asarray(observation_mask).astype(bool)
    residual_target = np.where(
        mask[train_positions] & np.isfinite(oof_anchor[train_positions]),
        truth[train_positions] - oof_anchor[train_positions], 0.0,
    ).astype(np.float32)
    residual_mask = (mask[train_positions] &
                     np.isfinite(oof_anchor[train_positions])).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(features[train_positions], dtype=np.float32)),
                      torch.from_numpy(residual_target), torch.from_numpy(residual_mask)),
        batch_size=config.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    control_positions = np.flatnonzero(
        train_rows & metadata["perturbation_no_concentration"].isin(CONTROL_NAMES).to_numpy())
    triples, interpolation_weights = make_time_triples(metadata, train_rows)
    control_features = torch.from_numpy(
        np.asarray(features[control_positions], dtype=np.float32)).to(device)
    triple_features = torch.from_numpy(
        np.asarray(features[triples], dtype=np.float32)).to(device) if len(triples) else None
    triple_weights = torch.from_numpy(interpolation_weights).to(device) if len(triples) else None

    model = PhysicsResidualCorrector(features.shape[1], truth.shape[1],
                                     config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config.seed)
    selection_positions = np.flatnonzero(selection_rows)
    selection_features = np.asarray(features[selection_positions], dtype=np.float32)
    selection_truth = truth[selection_positions]
    selection_mask = mask[selection_positions] & np.isfinite(selection_truth)
    anchor_selection = validation_anchor[selection_positions]
    baseline_error = anchor_selection - selection_truth
    baseline_mse = float(np.where(selection_mask, baseline_error * baseline_error, 0.0).sum()
                         / max(selection_mask.sum(), 1))
    best_rmse = float(np.sqrt(baseline_mse))
    best_epoch = 0
    best_state = copy.deepcopy({key: value.detach().cpu()
                                for key, value in model.state_dict().items()})
    stale = 0
    history: list[dict[str, float | int]] = [{
        "epoch": 0, "train_main": np.nan, "train_control": 0.0,
        "train_curvature": 0.0, "selection_log2_rmse": best_rmse,
    }]

    for epoch in range(1, config.epochs + 1):
        model.train()
        main_values, control_values, curvature_values = [], [], []
        for batch_features, batch_target, batch_mask in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                correction = model(batch_features)
                main_loss = _masked_huber(correction, batch_target, batch_mask)
                control_loss = correction.sum() * 0.0
                if config.control_weight > 0 and len(control_features):
                    selected = rng.integers(len(control_features),
                                            size=min(config.auxiliary_batch_size, len(control_features)))
                    control_correction = model(control_features[selected])
                    control_loss = F.huber_loss(control_correction, torch.zeros_like(control_correction))
                curvature_loss = correction.sum() * 0.0
                if config.curvature_weight > 0 and triple_features is not None:
                    selected = rng.integers(len(triple_features),
                                            size=min(config.auxiliary_batch_size, len(triple_features)))
                    triple_correction = model(triple_features[selected].reshape(-1, features.shape[1]))
                    triple_correction = triple_correction.reshape(len(selected), 3, -1)
                    weights = triple_weights[selected]
                    interpolated = (weights[:, :1] * triple_correction[:, 0] +
                                    weights[:, 1:] * triple_correction[:, 2])
                    curvature = triple_correction[:, 1] - interpolated
                    curvature_loss = F.huber_loss(curvature, torch.zeros_like(curvature))
                loss = (main_loss + config.control_weight * control_loss +
                        config.curvature_weight * curvature_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            main_values.append(float(main_loss.item()))
            control_values.append(float(control_loss.item()))
            curvature_values.append(float(curvature_loss.item()))

        correction = predict_residuals(model, selection_features, device, config.batch_size)
        prediction = anchor_selection + correction
        error = prediction - selection_truth
        mse = float(np.where(selection_mask, error * error, 0.0).sum()
                    / max(selection_mask.sum(), 1))
        rmse = float(np.sqrt(mse))
        history.append({
            "epoch": epoch, "train_main": float(np.mean(main_values)),
            "train_control": float(np.mean(control_values)),
            "train_curvature": float(np.mean(curvature_values)),
            "selection_log2_rmse": rmse,
        })
        if rmse < best_rmse - 1e-6:
            best_rmse, best_epoch = rmse, epoch
            best_state = copy.deepcopy({key: value.detach().cpu()
                                        for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state); model.to(device)
    all_correction = predict_residuals(model, np.asarray(features, dtype=np.float32),
                                       device, config.batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "config": asdict(config),
        "best_epoch": best_epoch, "best_selection_log2_rmse": best_rmse,
        "time_triples": len(triples), "control_samples": len(control_positions),
        "input_dim": features.shape[1], "output_dim": truth.shape[1],
    }, output_dir / "physics_residual.pt")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    info = {
        "config": asdict(config), "device": str(device),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch, "best_selection_log2_rmse": best_rmse,
        "baseline_selection_log2_rmse": float(np.sqrt(baseline_mse)),
        "time_triples": len(triples), "control_samples": len(control_positions),
    }
    return all_correction.astype(np.float32), history_frame, info
