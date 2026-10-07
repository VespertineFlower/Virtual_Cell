from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MaskedLossConfig:
    kind: str = "pooled_mse"
    huber_delta: float = 1.0
    std_floor: float = 0.5
    std_ceiling: float = 2.0

    def __post_init__(self) -> None:
        allowed = {"pooled_mse", "protein_balanced_mse", "variance_standardized_mse", "masked_huber"}
        if self.kind not in allowed:
            raise ValueError(f"Unsupported masked loss: {self.kind}")
        if self.huber_delta <= 0 or self.std_floor <= 0 or self.std_ceiling < self.std_floor:
            raise ValueError("Invalid masked loss hyperparameters")


@dataclass(frozen=True)
class MaskedLossState:
    protein_weights: np.ndarray
    protein_counts: np.ndarray
    protein_std: np.ndarray
    config: MaskedLossConfig

    def summary(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "count_quantiles": np.quantile(self.protein_counts, [0, .1, .5, .9, 1]).tolist(),
            "std_quantiles": np.quantile(self.protein_std, [0, .1, .5, .9, 1]).tolist(),
            "weight_quantiles": np.quantile(self.protein_weights, [0, .1, .5, .9, 1]).tolist(),
        }


def fit_masked_loss_state(y: np.ndarray, mask: np.ndarray, train_rows: np.ndarray,
                          config: MaskedLossConfig) -> MaskedLossState:
    values = np.asarray(y)[train_rows]
    observed = np.asarray(mask)[train_rows].astype(bool)
    counts = observed.sum(axis=0).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    sums = np.where(observed, values, 0.0).sum(axis=0, dtype=np.float64)
    squared = np.where(observed, values * values, 0.0).sum(axis=0, dtype=np.float64)
    means = sums / safe_counts
    std = np.sqrt(np.maximum(squared / safe_counts - means * means, 0.0))
    weights = np.ones_like(std)
    if config.kind == "protein_balanced_mse":
        active = counts > 0
        weights[active] = counts[active].sum() / (active.sum() * counts[active])
        weights[~active] = 0.0
    elif config.kind == "variance_standardized_mse":
        clipped = np.clip(std, config.std_floor, config.std_ceiling)
        weights = 1.0 / np.square(clipped)
    return MaskedLossState(weights.astype(np.float32), counts.astype(np.float32),
                           std.astype(np.float32), config)


def masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                protein_weights: torch.Tensor, config: MaskedLossConfig) -> torch.Tensor:
    if config.kind == "masked_huber":
        elementwise = F.huber_loss(prediction, target, reduction="none", delta=config.huber_delta)
    else:
        elementwise = (prediction - target).square()
    weighted_mask = mask * protein_weights.reshape(1, -1)
    return (elementwise * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def masked_rowwise_pearson_loss(prediction: torch.Tensor, target: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
    """Mean 1-Pearson over rows, using only jointly observed protein values."""
    counts = mask.sum(dim=1)
    safe = counts.clamp_min(1.0)
    pred_mean = (prediction * mask).sum(dim=1) / safe
    target_mean = (target * mask).sum(dim=1) / safe
    pred_centered = (prediction - pred_mean[:, None]) * mask
    target_centered = (target - target_mean[:, None]) * mask
    numerator = (pred_centered * target_centered).sum(dim=1)
    variance_product = (pred_centered.square().sum(dim=1) *
                        target_centered.square().sum(dim=1))
    denominator = torch.sqrt(variance_product + 1e-8)
    valid = (counts >= 2) & (variance_product > 1e-8)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return (1.0 - numerator[valid] / denominator[valid]).mean()


def masked_objective_numpy(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray,
                           state: MaskedLossState) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(target)
    error = np.asarray(prediction) - np.asarray(target)
    if state.config.kind == "masked_huber":
        absolute = np.abs(error)
        delta = state.config.huber_delta
        elementwise = np.where(absolute <= delta, 0.5 * error * error,
                               delta * (absolute - 0.5 * delta))
    else:
        elementwise = error * error
    weights = state.protein_weights.reshape(1, -1)
    denominator = np.where(valid, weights, 0.0).sum(dtype=np.float64)
    numerator = np.where(valid, elementwise * weights, 0.0).sum(dtype=np.float64)
    return float(numerator / max(denominator, 1.0))
