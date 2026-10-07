from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import MaskedLossConfig, fit_masked_loss_state, masked_loss, masked_objective_numpy
from .mlp import set_reproducible_seed


MIXED_FIELDS = (
    "Strains", "perturbation_no_concentration", "Medium", "Temperature",
    "data_source", "instrument", "Yeast_cell_plate", "pert_time_unit",
)


@dataclass(frozen=True)
class LowRankMixedConfig:
    rank: int = 64
    batch_size: int = 128
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    interaction_scale: float = 0.20
    latent_penalty: float = 1e-4
    seed: int = 20260804


class MixedEffectEncoder:
    def __init__(self, vocabularies: dict[str, list[str]], time_max: float) -> None:
        self.vocabularies = vocabularies
        self.time_max = max(float(time_max), 1.0)

    @classmethod
    def fit(cls, metadata: pd.DataFrame) -> "MixedEffectEncoder":
        vocabularies = {
            field: sorted(metadata[field].fillna("<NA>").astype(str).unique().tolist())
            for field in MIXED_FIELDS
        }
        time = pd.to_numeric(metadata["pert_time"], errors="coerce").fillna(0.0)
        return cls(vocabularies, float(time.max()))

    def transform(self, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        categorical = np.zeros((len(metadata), len(MIXED_FIELDS)), dtype=np.int64)
        for column, field in enumerate(MIXED_FIELDS):
            lookup = {value: position + 1 for position, value in enumerate(self.vocabularies[field])}
            categorical[:, column] = [
                lookup.get(value, 0)
                for value in metadata[field].fillna("<NA>").astype(str)
            ]
        raw_time = pd.to_numeric(metadata["pert_time"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        time = np.stack((raw_time / self.time_max,
                         np.log1p(raw_time) / np.log1p(self.time_max)), axis=1).astype(np.float32)
        return categorical, time

    def summary(self) -> dict[str, object]:
        return {"fields": list(MIXED_FIELDS), "vocabularies": self.vocabularies,
                "time_max": self.time_max, "unknown_index": 0}


class LowRankMixedEffects(nn.Module):
    def __init__(self, cardinalities: list[int], protein_dim: int,
                 rank: int, interaction_scale: float) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, rank, padding_idx=0)
            for cardinality in cardinalities
        ])
        self.time_network = nn.Sequential(nn.Linear(2, rank), nn.Tanh())
        self.global_effect = nn.Parameter(torch.zeros(rank))
        self.protein_loadings = nn.Parameter(torch.empty(rank, protein_dim))
        self.interaction_scale = interaction_scale
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        nn.init.normal_(self.protein_loadings, std=0.01)
        final = self.time_network[0]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, std=0.02)
        nn.init.zeros_(final.bias)

    def condition_latent(self, categorical: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        effects = [embedding(categorical[:, index])
                   for index, embedding in enumerate(self.embeddings)]
        time_effect = self.time_network(time)
        latent = self.global_effect + torch.stack(effects, dim=0).sum(dim=0) + time_effect
        strain, chemical = effects[0], effects[1]
        interactions = strain * chemical + strain * time_effect + chemical * time_effect
        return latent + self.interaction_scale * interactions

    def forward(self, categorical: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.condition_latent(categorical, time)
        return latent @ self.protein_loadings, latent


@torch.no_grad()
def predict_mixed(model: LowRankMixedEffects, categorical: np.ndarray, time: np.ndarray,
                  protein_mean: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(categorical), batch_size):
        cat = torch.from_numpy(categorical[start:start + batch_size]).to(device)
        local_time = torch.from_numpy(time[start:start + batch_size]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            residual, _ = model(cat, local_time)
        output.append(residual.float().cpu().numpy())
    return (np.concatenate(output) + protein_mean.reshape(1, -1)).astype(np.float32)


def train_low_rank_mixed(metadata: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
                         train_rows: np.ndarray, validation_rows: np.ndarray,
                         protein_mean: np.ndarray, config: LowRankMixedConfig,
                         output_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    set_reproducible_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MixedEffectEncoder.fit(metadata.loc[train_rows])
    categorical, time = encoder.transform(metadata)
    train_positions = np.flatnonzero(train_rows)
    validation_positions = np.flatnonzero(validation_rows)
    truth = np.asarray(y)[train_positions]
    observed = np.asarray(mask)[train_positions]
    target = np.where(observed, truth - protein_mean.reshape(1, -1), 0.0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(categorical[train_positions]), torch.from_numpy(time[train_positions]),
        torch.from_numpy(target), torch.from_numpy(observed.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(config.seed))
    model = LowRankMixedEffects(
        [len(encoder.vocabularies[field]) for field in MIXED_FIELDS], y.shape[1],
        config.rank, config.interaction_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_config = MaskedLossConfig("masked_huber")
    state = fit_masked_loss_state(y, mask, train_rows, loss_config)
    weights = torch.from_numpy(state.protein_weights).to(device)
    valid_truth = np.asarray(y)[validation_positions]
    valid_mask = np.asarray(mask)[validation_positions]
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        for cat, local_time, local_target, local_mask in loader:
            cat, local_time, local_target, local_mask = [
                value.to(device, non_blocking=True)
                for value in (cat, local_time, local_target, local_mask)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                prediction, latent = model(cat, local_time)
                loss = masked_loss(prediction, local_target, local_mask, weights, loss_config)
                loss = loss + config.latent_penalty * latent.square().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        validation_prediction = predict_mixed(
            model, categorical[validation_positions], time[validation_positions],
            protein_mean, device, config.batch_size)
        objective = masked_objective_numpy(validation_prediction, valid_truth, valid_mask, state)
        valid = valid_mask & np.isfinite(valid_truth)
        rmse = float(np.sqrt(np.where(valid, (validation_prediction - valid_truth) ** 2, 0.0).sum()
                             / valid.sum()))
        history.append({"epoch": epoch, "validation_objective": objective,
                        "validation_log2_rmse": rmse})
        if objective < best_loss - 1e-6:
            best_loss, best_epoch, stale = objective, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu()
                                        for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Low-rank mixed model did not produce a checkpoint")
    model.load_state_dict(best_state)
    prediction = predict_mixed(model, categorical[validation_positions], time[validation_positions],
                               protein_mean, device, config.batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = {"state_dict": best_state, "config": asdict(config),
                  "encoder": encoder.summary(), "best_epoch": best_epoch,
                  "best_validation_objective": best_loss,
                  "protein_mean": protein_mean, "protein_dim": int(y.shape[1])}
    torch.save(checkpoint, output_dir / "model.pt")
    info = {"best_epoch": best_epoch, "best_validation_objective": best_loss,
            "epochs_ran": len(history), "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "device": str(device), "config": asdict(config), "encoder": encoder.summary()}
    return prediction, info

