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

from .mlp import MLPConfig, set_reproducible_seed
from .losses import (MaskedLossConfig, fit_masked_loss_state, masked_loss,
                     masked_objective_numpy)


@dataclass(frozen=True)
class ContentAblation:
    name: str
    keep_strain_id: bool
    keep_chemical_id: bool
    use_strain_content: bool
    use_chemical_content: bool


ABLATIONS = (
    ContentAblation("chemical_content", True, False, False, True),
    ContentAblation("strain_content", False, True, True, False),
    ContentAblation("dual_content", False, False, True, True),
    ContentAblation("hybrid_id_content", True, True, True, True),
)


class ContentConditionMLP(nn.Module):
    """Condition MLP with optional learned projections of entity content."""

    def __init__(self, base_dim: int, output_dim: int, hidden_dim: int, dropout: float,
                 chemical_dim: int = 0, strain_dim: int = 0,
                 chemical_projection_dim: int = 128, strain_projection_dim: int = 64) -> None:
        super().__init__()
        self.chemical_projection = self._projector(chemical_dim, chemical_projection_dim)
        self.strain_projection = self._projector(strain_dim, strain_projection_dim)
        merged_dim = base_dim
        if self.chemical_projection is not None:
            merged_dim += chemical_projection_dim
        if self.strain_projection is not None:
            merged_dim += strain_projection_dim
        self.network = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _projector(input_dim: int, output_dim: int) -> nn.Module | None:
        if input_dim == 0:
            return None
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(self, base: torch.Tensor, chemical: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
        blocks = [base]
        if self.chemical_projection is not None:
            blocks.append(self.chemical_projection(chemical))
        if self.strain_projection is not None:
            blocks.append(self.strain_projection(strain))
        return self.network(torch.cat(blocks, dim=1))


def selected_base_columns(input_dim: int, slices: dict[str, slice], keep_strain_id: bool,
                          keep_chemical_id: bool) -> np.ndarray:
    removed: set[int] = set()
    if not keep_strain_id:
        block = slices["Strains"]
        removed.update(range(block.start, block.stop))
    if not keep_chemical_id:
        block = slices["perturbation_no_concentration"]
        removed.update(range(block.start, block.stop))
    return np.asarray([index for index in range(input_dim) if index not in removed], dtype=np.int64)


def selected_strain_columns(feature_names: np.ndarray) -> np.ndarray:
    """Select shared natural-isolate features, excluding the disjoint S288C-only k-mer block."""
    names = np.asarray(feature_names).astype(str)
    keep = np.fromiter(
        (name.startswith("population_") or name.startswith("peter2018_") for name in names),
        dtype=bool, count=len(names),
    )
    return np.flatnonzero(keep)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def _predict(model: nn.Module, base: np.ndarray, chemical: np.ndarray, strain: np.ndarray,
             device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(base), batch_size):
        inputs = (
            torch.from_numpy(base[start:start + batch_size]).to(device),
            torch.from_numpy(chemical[start:start + batch_size]).to(device),
            torch.from_numpy(strain[start:start + batch_size]).to(device),
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(*inputs)
        output.append(prediction.float().cpu().numpy())
    return np.concatenate(output, axis=0)


def train_content_mlp(base: np.ndarray, chemical: np.ndarray, strain: np.ndarray,
                      y_log2: np.ndarray, observation_mask: np.ndarray,
                      train_rows: np.ndarray, validation_rows: np.ndarray,
                      protein_mean_vector: np.ndarray, config: MLPConfig,
                      output_dir: Path, model_contract: dict[str, Any],
                      loss_config: MaskedLossConfig | None = None,
                      save_checkpoint: bool = True) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    set_reproducible_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = loss_config or MaskedLossConfig()
    loss_state = fit_masked_loss_state(y_log2, observation_mask, train_rows, loss_config)
    protein_weights = torch.from_numpy(loss_state.protein_weights).to(device)
    train_positions = np.flatnonzero(train_rows)
    validation_positions = np.flatnonzero(validation_rows)
    base = np.ascontiguousarray(base, dtype=np.float32)
    chemical = np.ascontiguousarray(chemical, dtype=np.float32)
    strain = np.ascontiguousarray(strain, dtype=np.float32)
    truth = np.asarray(y_log2)[train_positions]
    observed = np.asarray(observation_mask)[train_positions]
    residual = np.where(observed, truth - protein_mean_vector.reshape(1, -1), 0.0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(base[train_positions]), torch.from_numpy(chemical[train_positions]),
        torch.from_numpy(strain[train_positions]), torch.from_numpy(residual),
        torch.from_numpy(observed.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(config.seed))
    model = ContentConditionMLP(base.shape[1], y_log2.shape[1], config.hidden_dim, config.dropout,
                                chemical.shape[1], strain.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    validation_truth = np.asarray(y_log2)[validation_positions]
    validation_mask = np.asarray(observation_mask)[validation_positions]
    best_loss, best_rmse, best_epoch, stale, best_state = float("inf"), float("inf"), 0, 0, None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_sum = train_count = 0.0
        for batch_base, batch_chemical, batch_strain, batch_target, batch_mask in loader:
            batch = [x.to(device, non_blocking=True) for x in
                     (batch_base, batch_chemical, batch_strain, batch_target, batch_mask)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(batch[0], batch[1], batch[2])
                loss = masked_loss(prediction, batch[3], batch[4], protein_weights, loss_config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = float(batch[4].sum().item())
            train_sum += float(loss.item()) * count
            train_count += count
        residual_prediction = _predict(model, base[validation_positions], chemical[validation_positions],
                                       strain[validation_positions], device, config.batch_size)
        absolute = residual_prediction + protein_mean_vector.reshape(1, -1)
        mse = masked_objective_numpy(absolute, validation_truth, validation_mask, loss_state)
        valid = validation_mask & np.isfinite(validation_truth)
        actual_mse = float(np.where(valid, (absolute - validation_truth) ** 2, 0.0).sum() / valid.sum())
        history.append({"epoch": epoch, "train_masked_mse": train_sum / max(train_count, 1.0),
                        "validation_objective": mse, "validation_log2_rmse": float(np.sqrt(actual_mse))})
        if mse < best_loss - 1e-6:
            best_loss, best_epoch, stale = mse, epoch, 0
            best_rmse = float(np.sqrt(actual_mse))
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Content MLP training did not create a checkpoint")
    model.load_state_dict(best_state)
    prediction = _predict(model, base[validation_positions], chemical[validation_positions],
                          strain[validation_positions], device, config.batch_size)
    prediction += protein_mean_vector.reshape(1, -1)
    history_frame = pd.DataFrame(history)
    if save_checkpoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": asdict(config), "model_contract": model_contract,
                    "base_dim": int(base.shape[1]), "chemical_dim": int(chemical.shape[1]),
                    "strain_dim": int(strain.shape[1]), "output_dim": int(y_log2.shape[1]),
                    "best_epoch": best_epoch, "best_validation_masked_mse": best_loss,
                    "protein_mean": protein_mean_vector, "loss": loss_state.summary()}, output_dir / "model.pt")
        history_frame.to_csv(output_dir / "training_history.csv", index=False)
    info = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "best_epoch": best_epoch, "best_validation_masked_mse": best_loss,
            "best_validation_log2_rmse": best_rmse, "best_validation_objective": best_loss,
            "epochs_ran": len(history),
            "dimensions": {"base": int(base.shape[1]), "chemical": int(chemical.shape[1]), "strain": int(strain.shape[1])},
            "config": asdict(config), "loss": loss_state.summary()}
    return prediction.astype(np.float32), history_frame, info
