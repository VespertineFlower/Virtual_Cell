import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from baseline5_es_ifm import masked_metrics, seed_everything


METADATA_PATH = "./data/WAYB_WAYC_metadata_train_val(1).csv"
PROTEOME_PATH = "./data/WAYB_WAYC_proteome_raw_train_val.csv"
OUTPUT_PATH = "prediction_es_ifm_v2.csv"
COMPARISON_PATH = "es_ifm_v2_comparison.csv"
DIAGNOSTICS_PATH = "es_ifm_v2_diagnostics.json"

CONTROL_NAMES = {"Water", "DMSO"}
EXCLUDED_TREATMENTS = CONTROL_NAMES | {"Quality Control"}
MATCH_COLUMNS = [
    "data_source", "instrument", "Yeast_cell_plate", "Strains",
    "Medium", "Temperature", "pert_time",
]
BIO_COLUMNS = ["Medium", "Temperature", "pert_time"]
OBS_COLUMNS = ["data_source", "instrument", "Yeast_cell_plate"]
SCENARIOS = {
    "新化合物": "val_chem_only",
    "新菌株": "val_strain_only",
    "双重未知": "val_both",
    "时间验证": "val_time",
}
BASELINE_PATHS = {
    "ES-IFM-lite": "prediction_es_ifm_lite.csv",
    "ES-IFM-advanced-Hybrid": "prediction_es_ifm_advanced_hybrid.csv",
}


@dataclass
class V2Config:
    latent_dim: int = 64
    hidden_dim: int = 160
    embedding_dim: int = 24
    interaction_rank: int = 8
    batch_size: int = 256
    max_selection_epochs: int = 55
    patience: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    endpoint_weight: float = 0.25
    component_weight: float = 0.15
    center_weight: float = 0.03
    orth_weight: float = 0.005
    ridge: float = 4.0
    quadrature_steps: int = 3
    inference_steps: int = 8
    seed: int = 20260813


def joined_key(metadata: pd.DataFrame, columns: list[str]) -> pd.Series:
    return metadata[columns].astype(str).agg("\x1f".join, axis=1)


class FrozenEncoder:
    def __init__(self, metadata: pd.DataFrame, fit_rows: np.ndarray):
        fit = metadata.iloc[fit_rows]
        self.strains = {v: i + 1 for i, v in enumerate(sorted(fit["Strains"].astype(str).unique()))}
        fit_drugs = fit.loc[
            ~fit["perturbation_no_concentration"].isin(EXCLUDED_TREATMENTS),
            "perturbation_no_concentration",
        ].astype(str)
        self.drugs = {v: i + 1 for i, v in enumerate(sorted(fit_drugs.unique()))}
        self.bio_categories = {column: sorted(fit[column].astype(str).unique()) for column in BIO_COLUMNS}
        self.obs_categories = {column: sorted(fit[column].astype(str).unique()) for column in OBS_COLUMNS}

    @staticmethod
    def _blocks(metadata: pd.DataFrame, categories: dict[str, list[str]]) -> np.ndarray:
        blocks = []
        for column, values in categories.items():
            series = metadata[column].astype(str)
            blocks.append(np.column_stack([(series == value).to_numpy() for value in values]).astype(np.float32))
        return np.concatenate(blocks, axis=1)

    def transform(self, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        strains = metadata["Strains"].astype(str).map(self.strains).fillna(0).to_numpy(np.int64)
        drugs = metadata["perturbation_no_concentration"].astype(str).map(self.drugs).fillna(0).to_numpy(np.int64)
        return strains, drugs, self._blocks(metadata, self.bio_categories), self._blocks(metadata, self.obs_categories)

    def summary(self) -> dict[str, int]:
        return {
            "known_strains": len(self.strains),
            "known_drugs": len(self.drugs),
            "bio_features": sum(map(len, self.bio_categories.values())),
            "observation_features": sum(map(len, self.obs_categories.values())),
        }


class ESIFMV2(nn.Module):
    def __init__(self, config: V2Config, encoder: FrozenEncoder):
        super().__init__()
        h = config.hidden_dim
        e = config.embedding_dim
        z = config.latent_dim
        k = config.interaction_rank
        self.latent_dim = z
        self.interaction_rank = k
        self.strain_embedding = nn.Embedding(len(encoder.strains) + 1, e, padding_idx=0)
        self.drug_embedding = nn.Embedding(len(encoder.drugs) + 1, e, padding_idx=0)
        self.state = nn.Sequential(nn.Linear(z + 1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.mean_head = nn.Linear(h, z)
        self.drug_head = nn.Sequential(nn.Linear(h + e, h), nn.SiLU(), nn.Linear(h, z))
        self.strain_head = nn.Sequential(nn.Linear(h + e, h), nn.SiLU(), nn.Linear(h, z))
        bio_dim = sum(map(len, encoder.bio_categories.values()))
        obs_dim = sum(map(len, encoder.obs_categories.values()))
        self.bio_head = nn.Sequential(nn.Linear(h + bio_dim, h), nn.SiLU(), nn.Linear(h, z))
        self.observation_head = nn.Sequential(
            nn.Linear(obs_dim, h // 2), nn.SiLU(), nn.Dropout(0.05), nn.Linear(h // 2, z)
        )
        self.alpha = nn.Linear(e, k, bias=False)
        self.beta = nn.Linear(e, k, bias=False)
        self.interaction_basis = nn.Linear(h, k * z)

    def forward(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        strain_id: torch.Tensor,
        drug_id: torch.Tensor,
        bio: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        state = self.state(torch.cat([z, tau], dim=1))
        strain = self.strain_embedding(strain_id)
        drug = self.drug_embedding(drug_id)
        mean_part = self.mean_head(state)
        drug_part = self.drug_head(torch.cat([state, drug], dim=1))
        strain_part = self.strain_head(torch.cat([state, strain], dim=1))
        bio_part = self.bio_head(torch.cat([state, bio], dim=1))
        observation_part = self.observation_head(observation)
        coefficients = self.alpha(strain) * self.beta(drug)
        basis = self.interaction_basis(state).view(-1, self.interaction_rank, self.latent_dim)
        interaction_part = (coefficients.unsqueeze(-1) * basis).sum(dim=1)
        total = mean_part + drug_part + strain_part + interaction_part + bio_part + observation_part
        return total, (mean_part, drug_part, strain_part, interaction_part, bio_part, observation_part)


def strict_control_prediction(
    metadata: pd.DataFrame,
    y_filled: pd.DataFrame,
    fit_mask: np.ndarray,
) -> pd.DataFrame:
    control = metadata["perturbation_no_concentration"].isin(CONTROL_NAMES).to_numpy()
    control_rows = np.flatnonzero(fit_mask & control)
    exact_keys = joined_key(metadata, MATCH_COLUMNS)
    exact_means = y_filled.iloc[control_rows].groupby(exact_keys.iloc[control_rows]).mean()
    prediction = exact_means.reindex(exact_keys).set_axis(metadata.index)

    context_columns = ["data_source", "instrument", "Medium", "Temperature", "pert_time"]
    context_keys = joined_key(metadata, context_columns)
    context_means = y_filled.iloc[control_rows].groupby(context_keys.iloc[control_rows]).mean()
    prediction = prediction.fillna(context_means.reindex(context_keys).set_axis(metadata.index))
    return prediction.fillna(y_filled.iloc[control_rows].mean(axis=0))


def pseudo_s3_masks(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    official_train = metadata["split_final"].eq("train")
    treated = ~metadata["perturbation_no_concentration"].isin(EXCLUDED_TREATMENTS)
    drugs = sorted(metadata.loc[official_train & treated, "perturbation_no_concentration"].unique())
    held_drugs = drugs[2::8]
    held_strain = "CGD"
    entity_fit = official_train & ~metadata["Strains"].eq(held_strain)
    entity_fit &= ~metadata["perturbation_no_concentration"].isin(held_drugs)
    pseudo_validation = official_train & treated
    pseudo_validation &= metadata["Strains"].eq(held_strain)
    pseudo_validation &= metadata["perturbation_no_concentration"].isin(held_drugs)
    return entity_fit.to_numpy(), pseudo_validation.to_numpy(), {
        "held_strain": held_strain,
        "held_drugs": held_drugs,
        "fit_samples": int(entity_fit.sum()),
        "validation_samples": int(pseudo_validation.sum()),
    }


def weighted_effect_targets(
    velocity: np.ndarray,
    strains: np.ndarray,
    drugs: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_strains = np.unique(strains)
    unique_drugs = np.unique(drugs)
    strain_lookup = {value: i for i, value in enumerate(unique_strains)}
    drug_lookup = {value: i for i, value in enumerate(unique_drugs)}
    design = np.zeros((len(velocity), 1 + len(unique_strains) + len(unique_drugs)), dtype=np.float64)
    design[:, 0] = 1.0
    design[np.arange(len(velocity)), 1 + np.array([strain_lookup[x] for x in strains])] = 1.0
    offset = 1 + len(unique_strains)
    design[np.arange(len(velocity)), offset + np.array([drug_lookup[x] for x in drugs])] = 1.0
    root_weight = np.sqrt(weights).reshape(-1, 1)
    weighted_design = design * root_weight
    weighted_target = velocity.astype(np.float64) * root_weight
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(weighted_design.T @ weighted_design + penalty, weighted_design.T @ weighted_target)

    strain_coefficients = coefficients[1:offset]
    drug_coefficients = coefficients[offset:]
    strain_coefficients -= np.average(strain_coefficients, axis=0)
    drug_coefficients -= np.average(drug_coefficients, axis=0)
    mu = np.average(velocity, axis=0, weights=weights)
    strain_target = strain_coefficients[[strain_lookup[x] for x in strains]]
    drug_target = drug_coefficients[[drug_lookup[x] for x in drugs]]
    mean_target = np.broadcast_to(mu, velocity.shape)
    interaction_target = velocity - mean_target - strain_target - drug_target
    return tuple(np.asarray(x, dtype=np.float32) for x in (mean_target, drug_target, strain_target, interaction_target))


def mean_abs_cosine(parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
    normalized = [nn.functional.normalize(part.flatten(), dim=0, eps=1e-6) for part in parts]
    values = [torch.abs(torch.dot(a, b)) for i, a in enumerate(normalized) for b in normalized[i + 1:]]
    return torch.stack(values).mean()


def integrate_model(
    model: ESIFMV2,
    z0: torch.Tensor,
    strain: torch.Tensor,
    drug: torch.Tensor,
    bio: torch.Tensor,
    observation: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    z = z0
    dt = 1.0 / steps
    for step in range(steps):
        tau = torch.full((len(z), 1), (step + 0.5) * dt, device=z.device)
        velocity, _ = model(z, tau, strain, drug, bio, observation)
        z = z + dt * velocity
    return z


@dataclass
class PreparedRun:
    metadata: pd.DataFrame
    y_log2: pd.DataFrame
    y_filled: pd.DataFrame
    observation_mask: np.ndarray
    pca: PCA
    control_prediction: pd.DataFrame
    z0_all: np.ndarray
    z1_all: np.ndarray
    encoder: FrozenEncoder
    encoded: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    fit_treated_rows: np.ndarray
    validation_rows: np.ndarray
    targets: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    reliability: np.ndarray


def prepare_run(
    metadata: pd.DataFrame,
    y_log2: pd.DataFrame,
    observation_mask: np.ndarray,
    fit_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: V2Config,
) -> PreparedRun:
    protein_mean = y_log2.iloc[np.flatnonzero(fit_mask)].mean(axis=0)
    y_filled = y_log2.fillna(protein_mean)
    pca = PCA(n_components=config.latent_dim, svd_solver="randomized", random_state=config.seed)
    pca.fit(y_filled.iloc[np.flatnonzero(fit_mask)].to_numpy(np.float32))
    z1_all = pca.transform(y_filled.to_numpy(np.float32)).astype(np.float32)
    control_prediction = strict_control_prediction(metadata, y_filled, fit_mask)
    z0_all = pca.transform(control_prediction.to_numpy(np.float32)).astype(np.float32)

    treated = ~metadata["perturbation_no_concentration"].isin(EXCLUDED_TREATMENTS).to_numpy()
    fit_treated_rows = np.flatnonzero(fit_mask & treated)
    validation_rows = np.flatnonzero(validation_mask & treated)
    encoder = FrozenEncoder(metadata, np.flatnonzero(fit_mask))
    encoded = encoder.transform(metadata)
    velocity = z1_all[fit_treated_rows] - z0_all[fit_treated_rows]
    reliability = observation_mask[fit_treated_rows].mean(axis=1).astype(np.float32)
    targets = weighted_effect_targets(
        velocity,
        encoded[0][fit_treated_rows],
        encoded[1][fit_treated_rows],
        reliability,
        config.ridge,
    )
    return PreparedRun(
        metadata, y_log2, y_filled, observation_mask, pca, control_prediction,
        z0_all, z1_all, encoder, encoded, fit_treated_rows, validation_rows, targets, reliability,
    )


def train_epoch(
    model: ESIFMV2,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: V2Config,
) -> dict[str, float]:
    model.train()
    totals = np.zeros(5, dtype=np.float64)
    count = 0
    for batch in loader:
        z0, z1, strain, drug, bio, observation, mean_t, drug_t, strain_t, interaction_t, reliability = batch
        target_velocity = z1 - z0
        tau = torch.rand(len(z0), 1)
        z_tau = (1.0 - tau) * z0 + tau * z1
        velocity, parts = model(z_tau, tau, strain, drug, bio, observation)
        fm_per_row = (velocity - target_velocity).square().mean(dim=1)
        fm_loss = (fm_per_row * reliability).sum() / reliability.sum().clamp_min(1e-6)

        integrated_parts = [torch.zeros_like(target_velocity) for _ in range(4)]
        integrated_total = torch.zeros_like(target_velocity)
        for step in range(config.quadrature_steps):
            path_tau = torch.full((len(z0), 1), (step + 0.5) / config.quadrature_steps)
            path_z = (1.0 - path_tau) * z0 + path_tau * z1
            path_velocity, path_parts = model(path_z, path_tau, strain, drug, bio, observation)
            integrated_total += path_velocity / config.quadrature_steps
            for index in range(4):
                integrated_parts[index] += path_parts[index] / config.quadrature_steps

        endpoint_loss = nn.functional.mse_loss(z0 + integrated_total, z1)
        component_loss = sum(
            nn.functional.mse_loss(prediction, target)
            for prediction, target in zip(integrated_parts, (mean_t, drug_t, strain_t, interaction_t))
        )
        mean_p, drug_p, strain_p, interaction_p, bio_p, observation_p = parts
        center_loss = (
            drug_p.mean(dim=0).square().mean()
            + strain_p.mean(dim=0).square().mean()
            + interaction_p.mean(dim=0).square().mean()
            + observation_p.mean(dim=0).square().mean()
        )
        orth_loss = mean_abs_cosine(parts)
        loss = (
            fm_loss
            + config.endpoint_weight * endpoint_loss
            + config.component_weight * component_loss
            + config.center_weight * center_loss
            + config.orth_weight * orth_loss
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals += np.array([loss.item(), fm_loss.item(), endpoint_loss.item(), component_loss.item(), orth_loss.item()]) * len(z0)
        count += len(z0)
    values = totals / max(count, 1)
    return dict(zip(("total", "fm", "endpoint", "component", "orth"), values.tolist()))


def build_loader(run: PreparedRun, config: V2Config) -> DataLoader:
    rows = run.fit_treated_rows
    strains, drugs, bio, observation = run.encoded
    dataset = TensorDataset(
        torch.tensor(run.z0_all[rows]), torch.tensor(run.z1_all[rows]),
        torch.tensor(strains[rows]), torch.tensor(drugs[rows]),
        torch.tensor(bio[rows]), torch.tensor(observation[rows]),
        *(torch.tensor(target) for target in run.targets),
        torch.tensor(run.reliability),
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)


def validation_rmse(model: ESIFMV2, run: PreparedRun, config: V2Config) -> float:
    rows = run.validation_rows
    strains, drugs, bio, observation = run.encoded
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), 512):
            batch_rows = rows[start:start + 512]
            prediction = integrate_model(
                model,
                torch.tensor(run.z0_all[batch_rows]),
                torch.tensor(strains[batch_rows]), torch.tensor(drugs[batch_rows]),
                torch.tensor(bio[batch_rows]), torch.tensor(observation[batch_rows]),
                config.inference_steps,
            )
            outputs.append(prediction.numpy())
    latent = np.vstack(outputs)
    protein_prediction = run.pca.inverse_transform(latent)
    truth = run.y_log2.iloc[rows].to_numpy()
    valid = np.isfinite(truth)
    squared_error = np.where(valid, (protein_prediction - truth) ** 2, 0.0)
    return float(np.sqrt(squared_error.sum() / valid.sum()))


def select_epoch(run: PreparedRun, config: V2Config) -> tuple[int, list[dict[str, float]]]:
    model = ESIFMV2(config, run.encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = build_loader(run, config)
    best_epoch = 1
    best_rmse = float("inf")
    stale = 0
    history = []
    for epoch in range(1, config.max_selection_epochs + 1):
        losses = train_epoch(model, loader, optimizer, config)
        rmse = validation_rmse(model, run, config)
        history.append({"epoch": epoch, **losses, "pseudo_s3_rmse": rmse})
        if rmse < best_rmse - 1e-5:
            best_rmse = rmse
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"selection epoch {epoch}: pseudo-S3 RMSE={rmse:.5f}, best={best_rmse:.5f}")
        if stale >= config.patience:
            break
    return best_epoch, history


def fit_full(run: PreparedRun, config: V2Config, epochs: int) -> ESIFMV2:
    seed_everything(config.seed)
    model = ESIFMV2(config, run.encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = build_loader(run, config)
    for epoch in range(1, epochs + 1):
        losses = train_epoch(model, loader, optimizer, config)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"full epoch {epoch}/{epochs}: total={losses['total']:.5f}, fm={losses['fm']:.5f}")
    return model


def predict_all(model: ESIFMV2, run: PreparedRun, config: V2Config) -> pd.DataFrame:
    strains, drugs, bio, observation = run.encoded
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(run.metadata), 512):
            end = min(start + 512, len(run.metadata))
            latent = integrate_model(
                model, torch.tensor(run.z0_all[start:end]),
                torch.tensor(strains[start:end]), torch.tensor(drugs[start:end]),
                torch.tensor(bio[start:end]), torch.tensor(observation[start:end]),
                config.inference_steps,
            )
            outputs.append(run.pca.inverse_transform(latent.numpy()))
    prediction = pd.DataFrame(np.vstack(outputs), index=run.metadata.index, columns=run.y_log2.columns)
    control = run.metadata["perturbation_no_concentration"].isin(CONTROL_NAMES)
    prediction.loc[control] = run.control_prediction.loc[control].to_numpy(np.float32)
    prediction.index.name = "sample_ID"
    return prediction


def main() -> None:
    config = V2Config()
    seed_everything(config.seed)
    metadata = pd.read_csv(METADATA_PATH).set_index("sample_ID")
    proteome = pd.read_csv(PROTEOME_PATH).set_index("sample_ID")
    official_train = metadata["split_final"].eq("train").to_numpy()
    keep = proteome.iloc[np.flatnonzero(official_train)].isna().mean(axis=0) < 0.697
    y_log2 = np.log2(proteome.loc[:, keep])
    observation_mask = y_log2.notna().to_numpy()

    selection_fit, pseudo_validation, pseudo_info = pseudo_s3_masks(metadata)
    print("pseudo-S3:", json.dumps(pseudo_info, ensure_ascii=False))
    selection_run = prepare_run(
        metadata, y_log2, observation_mask, selection_fit, pseudo_validation, config
    )
    best_epoch, history = select_epoch(selection_run, config)
    print(f"selected full-training epochs: {best_epoch}")

    empty_validation = np.zeros(len(metadata), dtype=bool)
    full_run = prepare_run(metadata, y_log2, observation_mask, official_train, empty_validation, config)
    model = fit_full(full_run, config, best_epoch)
    prediction = predict_all(model, full_run, config)
    prediction.to_csv(OUTPUT_PATH)

    baseline_predictions = {
        name: pd.read_csv(path, index_col="sample_ID", usecols=["sample_ID", *y_log2.columns])
        for name, path in BASELINE_PATHS.items()
    }
    rows = []
    for scenario, split in SCENARIOS.items():
        ids = metadata.index[
            metadata["split_final"].eq(split)
            & ~metadata["perturbation_no_concentration"].isin(EXCLUDED_TREATMENTS)
        ]
        truth = y_log2.loc[ids]
        for name, values in [*baseline_predictions.items(), ("ES-IFM-v2", prediction)]:
            rmse, global_r2, protein_r2 = masked_metrics(truth, values.loc[ids])
            rows.append({
                "场景": scenario, "模型": name, "样本数": len(ids),
                "log2 RMSE": rmse, "Global R²": global_r2, "蛋白 R² 中位数": protein_r2,
            })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    diagnostics = {
        "config": asdict(config),
        "pseudo_s3": pseudo_info,
        "selected_epoch": best_epoch,
        "selection_history": history,
        "selection_encoder": selection_run.encoder.summary(),
        "full_encoder": full_run.encoder.summary(),
        "explained_variance_ratio_sum": float(full_run.pca.explained_variance_ratio_.sum()),
    }
    Path(DIAGNOSTICS_PATH).write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.rename(columns={"Global R²": "Global R2", "蛋白 R² 中位数": "蛋白 R2 中位数"}).to_string(index=False))
    print(f"saved: {OUTPUT_PATH}")
    print(f"saved: {COMPARISON_PATH}")
    print(f"saved: {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
