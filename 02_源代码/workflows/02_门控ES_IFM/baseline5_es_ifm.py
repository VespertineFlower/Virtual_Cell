import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


METADATA_PATH = "./data/WAYB_WAYC_metadata_train_val(1).csv"
PROTEOME_PATH = "./data/WAYB_WAYC_proteome_raw_train_val.csv"
OUTPUT_PATH = "prediction_es_ifm_lite.csv"
COMPARISON_PATH = "es_ifm_comparison.csv"

CONTROL_NAMES = {"Water", "DMSO"}
MATCH_COLUMNS = [
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
]
CONTEXT_COLUMNS = ["data_source", "instrument", "Medium", "Temperature", "pert_time"]
SCENARIOS = {
    "双重未知": "val_both",
    "新化合物": "val_chem_only",
    "新菌株": "val_strain_only",
    "时间验证": "val_time",
}

SEED = 42
N_COMPONENTS = 64
HIDDEN = 128
EMBED_DIM = 24
INTERACTION_RANK = 8
EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
ODE_STEPS = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_keys(metadata: pd.DataFrame) -> pd.Series:
    return metadata[MATCH_COLUMNS].astype(str).agg("\x1f".join, axis=1)


def fit_pca(y: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = y.mean(axis=0)
    centered = y - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].astype(np.float32)
    scores = centered @ components.T
    return mean.astype(np.float32), components, scores.astype(np.float32)


def encode_context(metadata: pd.DataFrame, train_mask: pd.Series) -> np.ndarray:
    train_values = metadata.loc[train_mask, CONTEXT_COLUMNS].astype(str)
    categories = {column: sorted(train_values[column].unique()) for column in CONTEXT_COLUMNS}
    blocks = []
    for column in CONTEXT_COLUMNS:
        values = metadata[column].astype(str)
        block = np.column_stack([(values == category).to_numpy() for category in categories[column]])
        blocks.append(block.astype(np.float32))
    return np.concatenate(blocks, axis=1)


def encode_ids(values: pd.Series, train_mask: pd.Series) -> tuple[np.ndarray, int]:
    # Index zero is reserved for a genuinely unseen condition.
    vocabulary = {value: i + 1 for i, value in enumerate(sorted(values.loc[train_mask].unique()))}
    encoded = values.map(vocabulary).fillna(0).astype(np.int64).to_numpy()
    return encoded, len(vocabulary) + 1


def masked_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> tuple[float, float, float]:
    mask = y_true.notna()
    true_values = y_true.where(mask)
    pred_values = y_pred.where(mask)
    residual = pred_values - true_values
    rmse = np.sqrt((residual**2).sum().sum() / mask.sum().sum())
    global_mean = true_values.stack().mean()
    global_r2 = 1.0 - (residual**2).sum().sum() / ((true_values - global_mean) ** 2).sum().sum()
    ss_res = (residual**2).sum(axis=0)
    ss_tot = ((true_values - true_values.mean(axis=0)) ** 2).sum(axis=0)
    valid = ss_tot > 0
    protein_r2 = (1.0 - ss_res[valid] / ss_tot[valid]).median()
    return float(rmse), float(global_r2), float(protein_r2)


def make_effect_targets(
    delta_scores: np.ndarray,
    strains: np.ndarray,
    drugs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = delta_scores.mean(axis=0, keepdims=True)
    drug_effect = np.zeros_like(delta_scores)
    strain_effect = np.zeros_like(delta_scores)

    for drug in np.unique(drugs):
        rows = drugs == drug
        drug_effect[rows] = delta_scores[rows].mean(axis=0) - mu
    for strain in np.unique(strains):
        rows = strains == strain
        strain_effect[rows] = delta_scores[rows].mean(axis=0) - mu

    interaction = delta_scores - mu - drug_effect - strain_effect
    return (
        np.repeat(mu, len(delta_scores), axis=0).astype(np.float32),
        drug_effect.astype(np.float32),
        strain_effect.astype(np.float32),
        interaction.astype(np.float32),
    )


class ESIFMLite(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, n_strains: int, n_drugs: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.strain_embedding = nn.Embedding(n_strains, EMBED_DIM, padding_idx=0)
        self.drug_embedding = nn.Embedding(n_drugs, EMBED_DIM, padding_idx=0)
        self.state = nn.Sequential(
            nn.Linear(latent_dim + 1, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(HIDDEN, latent_dim)
        self.drug_head = nn.Sequential(nn.Linear(HIDDEN + EMBED_DIM, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, latent_dim))
        self.strain_head = nn.Sequential(nn.Linear(HIDDEN + EMBED_DIM, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, latent_dim))
        self.context_head = nn.Sequential(nn.Linear(HIDDEN + context_dim, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, latent_dim))
        self.alpha = nn.Linear(EMBED_DIM, INTERACTION_RANK, bias=False)
        self.beta = nn.Linear(EMBED_DIM, INTERACTION_RANK, bias=False)
        self.interaction_basis = nn.Linear(HIDDEN, INTERACTION_RANK * latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        strain_id: torch.Tensor,
        drug_id: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        state = self.state(torch.cat([z, tau], dim=1))
        strain = self.strain_embedding(strain_id)
        drug = self.drug_embedding(drug_id)
        mean_part = self.mean_head(state)
        drug_part = self.drug_head(torch.cat([state, drug], dim=1))
        strain_part = self.strain_head(torch.cat([state, strain], dim=1))
        context_part = self.context_head(torch.cat([state, context], dim=1))
        coefficients = self.alpha(strain) * self.beta(drug)
        basis = self.interaction_basis(state).view(-1, INTERACTION_RANK, self.latent_dim)
        interaction_part = (coefficients.unsqueeze(-1) * basis).sum(dim=1)
        total = mean_part + drug_part + strain_part + context_part + interaction_part
        return total, (mean_part, drug_part, strain_part, interaction_part, context_part)


@dataclass
class PreparedData:
    metadata: pd.DataFrame
    y_log2: pd.DataFrame
    protein_mean: pd.Series
    control_prediction: pd.DataFrame
    context: np.ndarray
    strain_ids: np.ndarray
    drug_ids: np.ndarray
    n_strains: int
    n_drugs: int
    paired_rows: np.ndarray
    delta_protein_mean: np.ndarray
    delta_components: np.ndarray
    delta_scores: np.ndarray


def prepare_data() -> PreparedData:
    metadata = pd.read_csv(METADATA_PATH).set_index("sample_ID")
    proteome = pd.read_csv(PROTEOME_PATH).set_index("sample_ID")
    train_mask = metadata["split_final"].eq("train")
    control_mask = metadata["perturbation_no_concentration"].isin(CONTROL_NAMES)

    keep = proteome.loc[train_mask].isna().mean(axis=0) < 0.697
    y_log2 = np.log2(proteome.loc[:, keep])
    protein_mean = y_log2.loc[train_mask].mean(axis=0)
    y_filled = y_log2.fillna(protein_mean)

    keys = build_keys(metadata)
    train_controls = metadata.index[train_mask & control_mask]
    control_means = y_filled.loc[train_controls].groupby(keys.loc[train_controls]).mean()

    # Exact training controls are used where available. Else use a context-specific
    # train-control mean, then the global train-control mean. No validation labels enter.
    control_prediction = control_means.reindex(keys).set_axis(metadata.index)
    context_group_columns = ["data_source", "instrument", "Medium", "Temperature", "pert_time"]
    train_control_context = metadata.loc[train_controls, context_group_columns].astype(str).agg("\x1f".join, axis=1)
    all_context_keys = metadata[context_group_columns].astype(str).agg("\x1f".join, axis=1)
    context_control_means = y_filled.loc[train_controls].groupby(train_control_context).mean()
    context_fallback = context_control_means.reindex(all_context_keys).set_axis(metadata.index)
    control_prediction = control_prediction.fillna(context_fallback)
    control_prediction = control_prediction.fillna(y_filled.loc[train_controls].mean(axis=0)).fillna(protein_mean)

    treated_train = train_mask & ~control_mask
    paired = treated_train & keys.isin(control_means.index)
    paired_rows = np.flatnonzero(paired.to_numpy())
    paired_controls = control_means.reindex(keys.loc[paired]).set_axis(metadata.index[paired])
    delta = y_filled.loc[paired] - paired_controls
    delta_mean, delta_components, delta_scores = fit_pca(delta.to_numpy(np.float32), N_COMPONENTS)

    context = encode_context(metadata, train_mask)
    strain_ids, n_strains = encode_ids(metadata["Strains"], train_mask)
    drug_ids, n_drugs = encode_ids(metadata["perturbation_no_concentration"], train_mask)

    return PreparedData(
        metadata=metadata,
        y_log2=y_log2,
        protein_mean=protein_mean,
        control_prediction=control_prediction,
        context=context,
        strain_ids=strain_ids,
        drug_ids=drug_ids,
        n_strains=n_strains,
        n_drugs=n_drugs,
        paired_rows=paired_rows,
        delta_protein_mean=delta_mean,
        delta_components=delta_components,
        delta_scores=delta_scores,
    )


def train_model(data: PreparedData) -> ESIFMLite:
    rows = data.paired_rows
    paired_strains = data.strain_ids[rows]
    paired_drugs = data.drug_ids[rows]
    mu, drug_target, strain_target, interaction_target = make_effect_targets(
        data.delta_scores, paired_strains, paired_drugs
    )

    dataset = TensorDataset(
        torch.tensor(data.delta_scores),
        torch.tensor(paired_strains),
        torch.tensor(paired_drugs),
        torch.tensor(data.context[rows]),
        torch.tensor(mu),
        torch.tensor(drug_target),
        torch.tensor(strain_target),
        torch.tensor(interaction_target),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ESIFMLite(N_COMPONENTS, data.context.shape[1], data.n_strains, data.n_drugs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for delta, strain, drug, context, mu_t, drug_t, strain_t, interaction_t in loader:
            tau = torch.rand(len(delta), 1)
            z_tau = tau * delta
            velocity, parts = model(z_tau, tau, strain, drug, context)
            mean_p, drug_p, strain_p, interaction_p, _ = parts
            fm_loss = nn.functional.mse_loss(velocity, delta)
            component_loss = (
                nn.functional.mse_loss(mean_p, mu_t)
                + nn.functional.mse_loss(drug_p, drug_t)
                + nn.functional.mse_loss(strain_p, strain_t)
                + nn.functional.mse_loss(interaction_p, interaction_t)
            )
            center_loss = drug_p.mean(dim=0).square().mean() + strain_p.mean(dim=0).square().mean()
            endpoint_velocity, _ = model(delta, torch.ones_like(tau), strain, drug, context)
            endpoint_loss = nn.functional.mse_loss(endpoint_velocity, delta)
            loss = fm_loss + 0.25 * endpoint_loss + 0.20 * component_loss + 0.05 * center_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(delta)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{EPOCHS}, loss: {total_loss / len(dataset):.5f}")
    return model


def predict_delta(model: ESIFMLite, data: PreparedData) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(data.metadata), 512):
            end = min(start + 512, len(data.metadata))
            strain = torch.tensor(data.strain_ids[start:end])
            drug = torch.tensor(data.drug_ids[start:end])
            context = torch.tensor(data.context[start:end])
            z = torch.zeros(end - start, N_COMPONENTS)
            step_size = 1.0 / ODE_STEPS
            for step in range(ODE_STEPS):
                tau = torch.full((end - start, 1), (step + 0.5) * step_size)
                velocity, _ = model(z, tau, strain, drug, context)
                z = z + step_size * velocity
            outputs.append(z.numpy())
    return np.vstack(outputs)


def main() -> None:
    seed_everything(SEED)
    data = prepare_data()
    print(f"paired train responses: {len(data.paired_rows)}")
    print(f"retained proteins: {data.y_log2.shape[1]}")
    print(f"latent components: {N_COMPONENTS}")
    print(f"context features: {data.context.shape[1]}")

    model = train_model(data)
    delta_scores = predict_delta(model, data)
    delta_prediction = delta_scores @ data.delta_components + data.delta_protein_mean
    prediction = data.control_prediction + delta_prediction

    control_mask = data.metadata["perturbation_no_concentration"].isin(CONTROL_NAMES).to_numpy()
    prediction.iloc[control_mask] = data.control_prediction.iloc[control_mask]
    prediction.index.name = "sample_ID"
    prediction.to_csv(OUTPUT_PATH)

    rows = []
    for scenario, split in SCENARIOS.items():
        ids = data.metadata.index[data.metadata["split_final"].eq(split)]
        y_true = data.y_log2.loc[ids]
        mean_prediction = pd.DataFrame(
            np.tile(data.protein_mean.to_numpy(), (len(ids), 1)), index=ids, columns=y_true.columns
        )
        for name, pred in [
            ("蛋白均值", mean_prediction),
            ("训练集对照（严格）", data.control_prediction.loc[ids]),
            ("ES-IFM-lite", prediction.loc[ids]),
        ]:
            rmse, global_r2, protein_r2 = masked_metrics(y_true, pred)
            rows.append(
                {
                    "场景": scenario,
                    "模型": name,
                    "样本数": len(ids),
                    "log2 RMSE": rmse,
                    "Global R²": global_r2,
                    "蛋白 R² 中位数": protein_r2,
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    print(comparison.rename(columns={"Global R²": "Global R2", "蛋白 R² 中位数": "蛋白 R2 中位数"}).to_string(index=False))
    print(f"saved: {OUTPUT_PATH}")
    print(f"saved: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
