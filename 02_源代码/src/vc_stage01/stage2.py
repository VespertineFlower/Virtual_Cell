from __future__ import annotations

import copy
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import CONTROL_NAMES, MATCH_KEYS
from .losses import (MaskedLossConfig, fit_masked_loss_state, masked_loss,
                     masked_objective_numpy)


@dataclass
class Stage2Config:
    latent_dim: int = 64
    hidden_dim: int = 256
    batch_size: int = 128
    additive_epochs: int = 60
    control_epochs: int = 35
    flow_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    ode_steps: int = 8
    endpoint_weight: float = 0.25
    seed: int = 20260805


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strict_matched_controls(
    metadata: pd.DataFrame,
    y_log2: np.ndarray,
    target_rows: np.ndarray,
    control_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match targets while explicitly restricting which controls may be read."""
    if len(metadata) != len(y_log2):
        raise ValueError("metadata/y_log2 row mismatch")
    if len(target_rows) != len(metadata) or len(control_pool) != len(metadata):
        raise ValueError("row masks must match metadata length")
    allowed = control_pool & metadata["perturbation_no_concentration"].isin(CONTROL_NAMES).to_numpy()
    groups: dict[tuple[object, ...], list[int]] = {}
    for position in np.flatnonzero(allowed):
        key = tuple(metadata.iloc[position][field] for field in MATCH_KEYS)
        groups.setdefault(key, []).append(int(position))
    rows: list[int] = []
    controls: list[np.ndarray] = []
    counts: list[int] = []
    values = np.asarray(y_log2)
    for position in np.flatnonzero(target_rows):
        key = tuple(metadata.iloc[position][field] for field in MATCH_KEYS)
        positions = groups.get(key)
        if not positions:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            control = np.nanmean(values[positions], axis=0).astype(np.float32)
        if np.isfinite(control).any():
            rows.append(int(position))
            controls.append(control)
            counts.append(len(positions))
    if not rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, y_log2.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return np.asarray(rows), np.stack(controls), np.asarray(counts)


def fit_masked_pca(
    y_log2: np.ndarray,
    train_rows: np.ndarray,
    protein_mean: np.ndarray,
    latent_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Fit randomized PCA with train-mean imputation on training rows only."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = np.asarray(y_log2)[train_rows]
    filled = np.where(np.isfinite(train), train, protein_mean.reshape(1, -1))
    centered = np.ascontiguousarray(filled - protein_mean.reshape(1, -1), dtype=np.float32)
    x = torch.from_numpy(centered).to(device)
    q = min(latent_dim, min(x.shape) - 1)
    _, singular_values, components = torch.pca_lowrank(x, q=q, center=False, niter=4)
    total_ss = float(x.square().sum().item())
    captured_ss = float(singular_values.square().sum().item())
    return components.float().cpu().numpy(), {
        "latent_dim": int(q),
        "explained_variance_fraction": captured_ss / max(total_ss, 1e-12),
        "fit_rows": int(train_rows.sum()),
        "fit_scope": "split_final=train only",
        "missing_policy": "training protein mean imputation",
    }


def encode_proteome(y: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(y), y, mean.reshape(1, -1))
    return np.asarray((filled - mean.reshape(1, -1)) @ components, dtype=np.float32)


def decode_proteome(z: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return np.asarray(z @ components.T + mean.reshape(1, -1), dtype=np.float32)


def feature_slices(encoder_info: dict[str, Any]) -> dict[str, slice]:
    offset = 0
    result: dict[str, slice] = {}
    for field in encoder_info["categorical_fields"]:
        width = len(encoder_info["vocabularies"][field])
        result[field] = slice(offset, offset + width)
        offset += width
    result["__continuous__"] = slice(offset, int(encoder_info["output_dim"]))
    return result


class AdditiveConditionModel(nn.Module):
    """No-interaction baseline: strain + compound + context main effects."""

    def __init__(self, input_dim: int, output_dim: int, slices: dict[str, slice]):
        super().__init__()
        self.strain_slice = slices["Strains"]
        self.drug_slice = slices["perturbation_no_concentration"]
        excluded = set(range(self.strain_slice.start, self.strain_slice.stop)) | set(
            range(self.drug_slice.start, self.drug_slice.stop)
        )
        self.register_buffer(
            "context_indices",
            torch.tensor([i for i in range(input_dim) if i not in excluded], dtype=torch.long),
        )
        self.strain = nn.Linear(self.strain_slice.stop - self.strain_slice.start, output_dim, bias=False)
        self.drug = nn.Linear(self.drug_slice.stop - self.drug_slice.start, output_dim, bias=False)
        self.context = nn.Linear(len(self.context_indices), output_dim, bias=True)
        for layer in (self.strain, self.drug, self.context):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.strain(x[:, self.strain_slice])
            + self.drug(x[:, self.drug_slice])
            + self.context(x.index_select(1, self.context_indices))
        )


class LatentMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConditionalVelocityField(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + condition_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, latent_dim),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, z: torch.Tensor, tau: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([z, tau, condition], dim=1))


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).square() * mask
    return error.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def _predict_additive(
    model: nn.Module, features: np.ndarray, mean: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start : start + batch_size]).to(device)
        output.append(model(x).float().cpu().numpy() + mean.reshape(1, -1))
    return np.concatenate(output).astype(np.float32)


def train_additive_model(
    features: np.ndarray,
    y_log2: np.ndarray,
    mask: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    mean: np.ndarray,
    encoder_info: dict[str, Any],
    config: Stage2Config,
    output_dir: Path,
    loss_config: MaskedLossConfig | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = loss_config or MaskedLossConfig()
    loss_state = fit_masked_loss_state(y_log2, mask, train_rows, loss_config)
    protein_weights = torch.from_numpy(loss_state.protein_weights).to(device)
    model = AdditiveConditionModel(features.shape[1], y_log2.shape[1], feature_slices(encoder_info)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    positions = np.flatnonzero(train_rows)
    target = np.asarray(y_log2)[positions]
    observed = np.asarray(mask)[positions]
    residual = np.where(observed, target - mean.reshape(1, -1), 0.0).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(features[positions])),
            torch.from_numpy(residual), torch.from_numpy(observed.astype(np.float32)),
        ),
        batch_size=config.batch_size, shuffle=True, num_workers=0,
    )
    valid_pos = np.flatnonzero(validation_rows)
    best_loss, best_rmse, best_epoch, stale, best_state = float("inf"), float("inf"), 0, 0, None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.additive_epochs + 1):
        model.train()
        for x, batch_target, batch_mask in loader:
            x, batch_target, batch_mask = x.to(device), batch_target.to(device), batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), batch_target, batch_mask, protein_weights, loss_config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        prediction = _predict_additive(model, features[valid_pos], mean, device, config.batch_size)
        truth = np.asarray(y_log2)[valid_pos]
        valid_mask = np.asarray(mask)[valid_pos] & np.isfinite(truth)
        mse = masked_objective_numpy(prediction, truth, valid_mask, loss_state)
        actual_mse = float(np.where(valid_mask, (prediction - truth) ** 2, 0.0).sum() / valid_mask.sum())
        history.append({"epoch": epoch, "validation_objective": mse,
                        "validation_log2_rmse": float(np.sqrt(actual_mse))})
        if mse < best_loss - 1e-6:
            best_loss, best_epoch, stale = mse, epoch, 0
            best_rmse = float(np.sqrt(actual_mse))
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("additive model failed to create a checkpoint")
    model.load_state_dict(best_state)
    prediction = _predict_additive(model, features, mean, device, config.batch_size)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "additive_history.csv", index=False)
    torch.save(
        {"state_dict": best_state, "config": asdict(config), "encoder": encoder_info,
         "protein_mean": mean, "input_dim": features.shape[1], "output_dim": y_log2.shape[1],
         "loss": loss_state.summary()},
        output_dir / "additive_model.pt",
    )
    return prediction, history_frame, {
        "best_epoch": best_epoch,
        "best_validation_log2_rmse": best_rmse, "best_validation_objective": best_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()), "device": str(device),
        "loss": loss_state.summary(),
    }


def make_control_context_features(encoder: Any, metadata: pd.DataFrame) -> np.ndarray:
    context = metadata.copy()
    context["perturbation_no_concentration"] = "__NO_PERTURBATION_INPUT__"
    return encoder.transform(context)


def _train_latent_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    config: Stage2Config,
    device: torch.device,
) -> tuple[LatentMLP, list[dict[str, float | int]], dict[str, Any]]:
    model = LatentMLP(features.shape[1], targets.shape[1], config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(targets)),
                        batch_size=config.batch_size, shuffle=True, num_workers=0)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history: list[dict[str, float | int]] = []
    vx, vy = torch.from_numpy(validation_features).to(device), torch.from_numpy(validation_targets).to(device)
    for epoch in range(1, config.control_epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(nn.functional.mse_loss(model(vx), vy).item())
        history.append({"epoch": epoch, "validation_latent_mse": valid_loss})
        if valid_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = valid_loss, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("control estimator failed to create a checkpoint")
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_validation_latent_mse": best_loss}


@torch.no_grad()
def integrate_flow(
    control_model: nn.Module,
    velocity: ConditionalVelocityField,
    control_features: np.ndarray,
    condition_features: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps: int,
    observed_z0: np.ndarray | None = None,
) -> np.ndarray:
    control_model.eval()
    velocity.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(condition_features), batch_size):
        condition = torch.from_numpy(condition_features[start : start + batch_size]).to(device)
        if observed_z0 is None:
            base_x = torch.from_numpy(control_features[start : start + batch_size]).to(device)
            z = control_model(base_x)
        else:
            z = torch.from_numpy(observed_z0[start : start + batch_size]).to(device)
        for step in range(steps):
            tau = torch.full((len(z), 1), (step + 0.5) / steps, device=device)
            z = z + velocity(z, tau, condition) / steps
        outputs.append(z.float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def train_conditional_flow(
    condition_features: np.ndarray,
    control_features: np.ndarray,
    pair_rows: np.ndarray,
    pair_z0: np.ndarray,
    pair_z1: np.ndarray,
    validation_rows: np.ndarray,
    validation_z1: np.ndarray,
    config: Stage2Config,
    output_dir: Path,
    pca_payload: dict[str, Any],
    encoder_info: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(pair_rows))
    cut = max(1, int(0.9 * len(order)))
    fit_idx, hold_idx = order[:cut], order[cut:]
    control_model, control_history, control_info = _train_latent_regressor(
        np.ascontiguousarray(control_features[pair_rows[fit_idx]], dtype=np.float32),
        np.ascontiguousarray(pair_z0[fit_idx], dtype=np.float32),
        np.ascontiguousarray(control_features[pair_rows[hold_idx]], dtype=np.float32),
        np.ascontiguousarray(pair_z0[hold_idx], dtype=np.float32), config, device,
    )
    velocity = ConditionalVelocityField(pair_z0.shape[1], condition_features.shape[1], config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(velocity.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(condition_features[pair_rows], dtype=np.float32)),
                      torch.from_numpy(np.ascontiguousarray(pair_z0, dtype=np.float32)),
                      torch.from_numpy(np.ascontiguousarray(pair_z1, dtype=np.float32))),
        batch_size=config.batch_size, shuffle=True, num_workers=0,
    )
    valid_positions = np.flatnonzero(validation_rows)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.flow_epochs + 1):
        velocity.train()
        total_loss, batches = 0.0, 0
        for condition, z0, z1 in loader:
            condition, z0, z1 = condition.to(device), z0.to(device), z1.to(device)
            tau = torch.rand((len(z0), 1), device=device)
            target_velocity = z1 - z0
            z_tau = (1.0 - tau) * z0 + tau * z1
            optimizer.zero_grad(set_to_none=True)
            fm_loss = nn.functional.mse_loss(velocity(z_tau, tau, condition), target_velocity)
            zero = torch.zeros((len(z0), 1), device=device)
            endpoint_loss = nn.functional.mse_loss(z0 + velocity(z0, zero, condition), z1)
            loss = fm_loss + config.endpoint_weight * endpoint_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(velocity.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        predicted_z1 = integrate_flow(
            control_model, velocity, control_features[valid_positions], condition_features[valid_positions],
            device, config.batch_size, config.ode_steps,
        )
        valid_mse = float(np.mean((predicted_z1 - validation_z1) ** 2))
        history.append({"epoch": epoch, "train_objective": total_loss / max(batches, 1),
                        "validation_latent_mse": valid_mse})
        if valid_mse < best_loss - 1e-6:
            best_loss, best_epoch, stale = valid_mse, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in velocity.state_dict().items()})
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("flow model failed to create a checkpoint")
    velocity.load_state_dict(best_state)
    all_z1 = integrate_flow(control_model, velocity, control_features, condition_features,
                            device, config.batch_size, config.ode_steps)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "flow_history.csv", index=False)
    pd.DataFrame(control_history).to_csv(output_dir / "control_estimator_history.csv", index=False)
    torch.save(
        {"control_state_dict": {k: v.detach().cpu() for k, v in control_model.state_dict().items()},
         "velocity_state_dict": best_state, "config": asdict(config),
         "condition_dim": condition_features.shape[1], "latent_dim": pair_z0.shape[1],
         "encoder": encoder_info, "pca": pca_payload},
        output_dir / "conditional_flow.pt",
    )
    return all_z1, history_frame, {
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_pairs": len(pair_rows), "best_epoch": best_epoch,
        "best_validation_latent_mse": best_loss, "ode_steps": config.ode_steps,
        "control_estimator": control_info,
        "control_parameter_count": sum(p.numel() for p in control_model.parameters()),
        "velocity_parameter_count": sum(p.numel() for p in velocity.parameters()),
    }
