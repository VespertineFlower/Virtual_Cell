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

from .losses import MaskedLossConfig, fit_masked_loss_state, masked_objective_numpy
from .mlp import ConditionMLP, MLPConfig, predict_residuals, set_reproducible_seed


@dataclass(frozen=True)
class ProteinGroupingConfig:
    method: str = "response"
    n_heads: int = 8
    signature_components: int = 16
    seed: int = 20260804

    def __post_init__(self) -> None:
        if self.method not in {"random", "missingness", "response"}:
            raise ValueError(f"Unsupported protein grouping method: {self.method}")
        if self.n_heads < 2:
            raise ValueError("n_heads must be at least two")


@dataclass
class MultiHeadMLPConfig(MLPConfig):
    n_heads: int = 8
    head_hidden_dim: int = 176
    shared_depth: int = 2
    head_balanced_loss: bool = True


def _lightweight_kmeans(values: np.ndarray, n_clusters: int, seed: int,
                        iterations: int = 30) -> np.ndarray:
    """Small deterministic k-means implementation for protein signatures."""
    rng = np.random.default_rng(seed)
    n_rows = values.shape[0]
    centers = [int(rng.integers(n_rows))]
    closest = np.square(values - values[centers[0]]).sum(axis=1)
    for _ in range(1, n_clusters):
        total = closest.sum()
        if total <= 1e-12:
            candidate = int(np.flatnonzero(~np.isin(np.arange(n_rows), centers))[0])
        else:
            candidate = int(rng.choice(n_rows, p=closest / total))
        centers.append(candidate)
        closest = np.minimum(closest, np.square(values - values[candidate]).sum(axis=1))
    centroids = values[np.asarray(centers)].copy()
    labels = np.zeros(n_rows, dtype=np.int64)
    for _ in range(iterations):
        distances = np.square(values[:, None, :] - centroids[None, :, :]).sum(axis=2)
        updated = distances.argmin(axis=1).astype(np.int64)
        # Empty clusters are repaired with points farthest from their assigned center.
        sizes = np.bincount(updated, minlength=n_clusters)
        for empty in np.flatnonzero(sizes == 0):
            candidates = np.flatnonzero(sizes[updated] > 1)
            move = candidates[np.argmax(distances[candidates, updated[candidates]])]
            sizes[updated[move]] -= 1
            updated[move] = empty
            sizes[empty] += 1
        if np.array_equal(updated, labels):
            break
        labels = updated
        centroids = np.stack([values[labels == cluster].mean(axis=0)
                              for cluster in range(n_clusters)])
    return labels


def make_protein_groups(
    y_log2: np.ndarray,
    observation_mask: np.ndarray,
    train_rows: np.ndarray,
    config: ProteinGroupingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign proteins to heads using training rows only."""
    observed = np.asarray(observation_mask)[train_rows].astype(bool)
    n_proteins = observed.shape[1]
    if config.n_heads > n_proteins:
        raise ValueError("n_heads cannot exceed the number of proteins")
    counts = observed.sum(axis=0)

    if config.method == "random":
        order = np.random.default_rng(config.seed).permutation(n_proteins)
        assignments = np.empty(n_proteins, dtype=np.int64)
        assignments[order] = np.arange(n_proteins) % config.n_heads
        signature_variance = None
    elif config.method == "missingness":
        # Rank bucketing is deterministic and keeps head sizes balanced while
        # placing proteins with similar observation counts together.
        order = np.argsort(counts, kind="stable")
        assignments = np.empty(n_proteins, dtype=np.int64)
        assignments[order] = np.minimum(
            np.arange(n_proteins) * config.n_heads // n_proteins,
            config.n_heads - 1,
        )
        signature_variance = None
    else:
        values = np.asarray(y_log2)[train_rows].astype(np.float32, copy=False)
        safe_counts = np.maximum(counts, 1)
        sums = np.where(observed, values, 0.0).sum(axis=0, dtype=np.float64)
        means = sums / safe_counts
        centered = np.where(observed, values - means.reshape(1, -1), 0.0)
        squared = np.square(centered, dtype=np.float32).sum(axis=0, dtype=np.float64)
        std = np.sqrt(np.maximum(squared / safe_counts, 1e-6))
        signatures = (centered / std.reshape(1, -1)).T.astype(np.float32)
        n_components = min(config.signature_components, signatures.shape[1])
        rng = np.random.default_rng(config.seed)
        projection = rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                                size=(signatures.shape[1], n_components))
        projection /= np.sqrt(signatures.shape[1])
        embedding = signatures @ projection
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True).clip(min=1e-6)
        # Break exact ties (e.g. invariant proteins) without affecting real distances.
        embedding += rng.normal(0.0, 1e-7, size=embedding.shape).astype(np.float32)
        assignments = _lightweight_kmeans(embedding, config.n_heads, config.seed)
        signature_variance = float(np.var(embedding, axis=0).sum())

    sizes = np.bincount(assignments, minlength=config.n_heads)
    if np.any(sizes == 0):
        raise RuntimeError(f"Protein grouping produced an empty head: {sizes.tolist()}")
    audit = {
        "config": asdict(config),
        "head_sizes": sizes.tolist(),
        "head_observation_count_means": [
            float(counts[assignments == head].mean()) for head in range(config.n_heads)
        ],
        "response_signature_variance": signature_variance,
    }
    return assignments, audit


class MultiHeadConditionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 head_hidden_dim: int, dropout: float, assignments: np.ndarray,
                 shared_depth: int = 2):
        super().__init__()
        assignments = np.asarray(assignments, dtype=np.int64)
        if assignments.shape != (output_dim,):
            raise ValueError("assignments must contain one head id per output protein")
        head_ids = np.unique(assignments)
        if not np.array_equal(head_ids, np.arange(len(head_ids))):
            raise ValueError("head ids must be contiguous from zero")
        if shared_depth not in {1, 2}:
            raise ValueError("shared_depth must be one or two")
        shared_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        if shared_depth == 2:
            shared_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.shared = nn.Sequential(*shared_layers)
        groups = [np.flatnonzero(assignments == head) for head in head_ids]
        order = np.concatenate(groups)
        inverse_order = np.argsort(order)
        self.register_buffer("inverse_order", torch.from_numpy(inverse_order).long())
        self.heads = nn.ModuleList()
        for group in groups:
            head = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, len(group)),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads.append(head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared = self.shared(features)
        grouped = torch.cat([head(shared) for head in self.heads], dim=1)
        return grouped.index_select(1, self.inverse_order)


def _head_balanced_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    assignments: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    elementwise = F.huber_loss(prediction, target, reduction="none", delta=delta)
    losses = []
    for head in range(int(assignments.max().item()) + 1):
        selected = assignments == head
        head_mask = mask[:, selected]
        losses.append((elementwise[:, selected] * head_mask).sum() /
                      head_mask.sum().clamp_min(1.0))
    return torch.stack(losses).mean()


def _head_balanced_huber_numpy(prediction: np.ndarray, target: np.ndarray,
                                mask: np.ndarray, assignments: np.ndarray,
                                delta: float) -> float:
    error = prediction - target
    absolute = np.abs(error)
    elementwise = np.where(absolute <= delta, 0.5 * error * error,
                           delta * (absolute - 0.5 * delta))
    losses = []
    for head in range(assignments.max() + 1):
        selected = assignments == head
        valid = mask[:, selected] & np.isfinite(target[:, selected])
        losses.append(float(np.where(valid, elementwise[:, selected], 0.0).sum(dtype=np.float64)
                            / max(valid.sum(), 1)))
    return float(np.mean(losses))


def parameter_matched_head_width(input_dim: int, output_dim: int, hidden_dim: int,
                                 n_heads: int, dropout: float = 0.1,
                                 shared_depth: int = 2) -> tuple[int, int, int]:
    baseline = ConditionMLP(input_dim, output_dim, hidden_dim, dropout)
    target = sum(parameter.numel() for parameter in baseline.parameters())
    # Multihead variable parameters are approximately h_head *
    # (output_dim + n_heads * (hidden_dim + 1)). Pick the closest practical width.
    shared = input_dim * hidden_dim + hidden_dim + 2 * hidden_dim
    if shared_depth == 2:
        shared += hidden_dim * hidden_dim + hidden_dim
    denominator = output_dim + n_heads * (hidden_dim + 1)
    width = max(16, int(round((target - shared - output_dim) / denominator / 8) * 8))
    probe_assignments = np.arange(output_dim) % n_heads
    model = MultiHeadConditionMLP(input_dim, output_dim, hidden_dim, width,
                                  dropout, probe_assignments, shared_depth=shared_depth)
    actual = sum(parameter.numel() for parameter in model.parameters())
    return width, target, actual


def train_multihead_condition_mlp(
    features: np.ndarray,
    y_log2: np.ndarray,
    observation_mask: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    protein_mean_vector: np.ndarray,
    assignments: np.ndarray,
    grouping_audit: dict[str, Any],
    config: MultiHeadMLPConfig,
    output_dir: Path,
    encoder_info: dict[str, Any],
    protein_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    set_reproducible_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = MaskedLossConfig("masked_huber", huber_delta=1.0)
    loss_state = fit_masked_loss_state(y_log2, observation_mask, train_rows, loss_config)
    assignment_tensor = torch.from_numpy(np.asarray(assignments)).long().to(device)

    train_positions = np.flatnonzero(train_rows)
    validation_positions = np.flatnonzero(validation_rows)
    train_features = np.ascontiguousarray(features[train_positions], dtype=np.float32)
    validation_features = np.ascontiguousarray(features[validation_positions], dtype=np.float32)
    train_truth = np.asarray(y_log2)[train_positions]
    train_mask = np.asarray(observation_mask)[train_positions]
    train_residual = np.where(train_mask, train_truth - protein_mean_vector[None, :], 0.0).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_residual),
                      torch.from_numpy(train_mask.astype(np.float32))),
        batch_size=config.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = MultiHeadConditionMLP(
        features.shape[1], y_log2.shape[1], config.hidden_dim,
        config.head_hidden_dim, config.dropout, assignments,
        shared_depth=config.shared_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    validation_truth = np.asarray(y_log2)[validation_positions]
    validation_mask = np.asarray(observation_mask)[validation_positions]
    best_loss = float("inf")
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch_features, batch_target, batch_mask in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                prediction = model(batch_features)
                if config.head_balanced_loss:
                    loss = _head_balanced_huber(prediction, batch_target, batch_mask,
                                                assignment_tensor, loss_config.huber_delta)
                else:
                    elementwise = F.huber_loss(prediction, batch_target, reduction="none",
                                               delta=loss_config.huber_delta)
                    loss = (elementwise * batch_mask).sum() / batch_mask.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            batches += 1

        residual = predict_residuals(model, validation_features, device, config.batch_size)
        absolute_prediction = residual + protein_mean_vector[None, :]
        if config.head_balanced_loss:
            validation_loss = _head_balanced_huber_numpy(
                absolute_prediction, validation_truth, validation_mask,
                assignments, loss_config.huber_delta)
        else:
            validation_loss = masked_objective_numpy(
                absolute_prediction, validation_truth, validation_mask, loss_state)
        valid = validation_mask & np.isfinite(validation_truth)
        mse = float(np.where(valid, (absolute_prediction - validation_truth) ** 2, 0.0).sum()
                    / max(valid.sum(), 1))
        rmse = float(np.sqrt(mse))
        history.append({"epoch": epoch, "train_objective": running_loss / max(batches, 1),
                        "validation_objective": validation_loss,
                        "validation_log2_rmse": rmse})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_rmse, best_epoch = validation_loss, rmse, epoch
            best_state = copy.deepcopy({key: value.detach().cpu()
                                        for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Multihead MLP training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    predictions = predict_residuals(model, features, device, config.batch_size)
    predictions += protein_mean_vector[None, :]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state, "config": asdict(config),
        "input_dim": int(features.shape[1]), "output_dim": int(y_log2.shape[1]),
        "assignments": assignments, "grouping": grouping_audit,
        "best_epoch": best_epoch, "best_validation_objective": best_loss,
        "best_validation_log2_rmse": best_rmse, "protein_mean": protein_mean_vector,
        "protein_names": protein_names, "encoder": encoder_info,
    }
    torch.save(checkpoint, output_dir / "multihead_condition_mlp.pt")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    info = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch, "best_validation_objective": best_loss,
        "best_validation_log2_rmse": best_rmse, "epochs_ran": len(history),
        "config": asdict(config), "grouping": grouping_audit,
    }
    return predictions.astype(np.float32), history_frame, info
