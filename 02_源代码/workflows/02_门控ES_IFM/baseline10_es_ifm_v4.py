import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baseline5_es_ifm import masked_metrics, seed_everything
from baseline8_es_ifm_v2 import (
    CONTROL_NAMES,
    EXCLUDED_TREATMENTS,
    METADATA_PATH,
    PROTEOME_PATH,
    SCENARIOS,
    ESIFMV2,
    PreparedRun,
    V2Config,
    mean_abs_cosine,
    prepare_run,
    pseudo_s3_masks,
)


EXTERNAL_MLP_DIR = "./outputs/stage1"
OUTPUT_TEMPLATE = "prediction_es_ifm_v4_{mode}.csv"
FLOW_OUTPUT_TEMPLATE = "prediction_es_ifm_v4_{mode}_no_residual.csv"
COMPARISON_TEMPLATE = "es_ifm_v4_{mode}_comparison.csv"
DIAGNOSTICS_TEMPLATE = "es_ifm_v4_{mode}_diagnostics.json"

EPISODIC_DRUGS = 5
RESIDUAL_LIMIT = 2.0
PROTEIN_ENDPOINT_WEIGHT = 0.35


class ESIFMV4(ESIFMV2):
    def __init__(self, config: V2Config, encoder, protein_dim: int):
        super().__init__(config, encoder)
        input_dim = (
            config.latent_dim
            + 2 * config.embedding_dim
            + sum(map(len, encoder.bio_categories.values()))
            + sum(map(len, encoder.obs_categories.values()))
        )
        self.residual_hidden = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        self.residual_output = nn.Linear(config.hidden_dim, protein_dim)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)
        self.residual_gate_logit = nn.Parameter(torch.tensor(-3.0))

    def protein_residual(
        self,
        latent: torch.Tensor,
        strain_id: torch.Tensor,
        drug_id: torch.Tensor,
        bio: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                latent,
                self.strain_embedding(strain_id),
                self.drug_embedding(drug_id),
                bio,
                observation,
            ],
            dim=1,
        )
        raw = self.residual_output(self.residual_hidden(features))
        return RESIDUAL_LIMIT * torch.sigmoid(self.residual_gate_logit) * torch.tanh(raw)


def episodic_ids(
    strains: torch.Tensor,
    drugs: torch.Tensor,
    epoch: int,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    if not enabled:
        return strains, drugs, {"held_strain": None, "held_drugs": []}
    unique_strains = torch.unique(strains[strains > 0]).sort().values
    unique_drugs = torch.unique(drugs[drugs > 0]).sort().values
    held_strain = unique_strains[(epoch - 1) % len(unique_strains)]
    offsets = (torch.arange(EPISODIC_DRUGS) * 7 + epoch - 1) % len(unique_drugs)
    held_drugs = unique_drugs[offsets]
    masked_strains = strains.clone()
    masked_drugs = drugs.clone()
    masked_strains[strains == held_strain] = 0
    for held_drug in held_drugs:
        masked_drugs[drugs == held_drug] = 0
    return masked_strains, masked_drugs, {
        "held_strain": int(held_strain.item()),
        "held_drugs": [int(value) for value in held_drugs.tolist()],
    }


def build_loader(run: PreparedRun, config: V2Config) -> DataLoader:
    rows = run.fit_treated_rows
    strains, drugs, bio, observation = run.encoded
    dataset = TensorDataset(
        torch.tensor(run.z0_all[rows]),
        torch.tensor(run.z1_all[rows]),
        torch.tensor(strains[rows]),
        torch.tensor(drugs[rows]),
        torch.tensor(bio[rows]),
        torch.tensor(observation[rows]),
        torch.tensor(run.y_filled.iloc[rows].to_numpy(np.float32)),
        torch.tensor(run.observation_mask[rows].astype(np.float32)),
        *(torch.tensor(target) for target in run.targets),
        torch.tensor(run.reliability),
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)


def train_epoch(
    model: ESIFMV4,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: V2Config,
    epoch: int,
    episodic: bool,
    pca_mean: torch.Tensor,
    pca_components: torch.Tensor,
) -> tuple[dict[str, float], dict[str, object]]:
    model.train()
    dataset = loader.dataset
    all_strains = dataset.tensors[2]
    all_drugs = dataset.tensors[3]
    _, _, episode = episodic_ids(all_strains, all_drugs, epoch, episodic)
    totals = np.zeros(6, dtype=np.float64)
    count = 0
    for batch in loader:
        (
            z0, z1, strain, drug, bio, observation, y_target, protein_mask,
            mean_t, drug_t, strain_t, interaction_t, reliability,
        ) = batch
        masked_strain = strain.clone()
        masked_drug = drug.clone()
        if episodic:
            masked_strain[strain == episode["held_strain"]] = 0
            for held_drug in episode["held_drugs"]:
                masked_drug[drug == held_drug] = 0

        target_velocity = z1 - z0
        tau = torch.rand(len(z0), 1)
        z_tau = (1.0 - tau) * z0 + tau * z1
        velocity, parts = model(z_tau, tau, masked_strain, masked_drug, bio, observation)
        fm_per_row = (velocity - target_velocity).square().mean(dim=1)
        fm_loss = (fm_per_row * reliability).sum() / reliability.sum().clamp_min(1e-6)

        integrated_parts = [torch.zeros_like(target_velocity) for _ in range(4)]
        integrated_total = torch.zeros_like(target_velocity)
        for step in range(config.quadrature_steps):
            path_tau = torch.full((len(z0), 1), (step + 0.5) / config.quadrature_steps)
            path_z = (1.0 - path_tau) * z0 + path_tau * z1
            path_velocity, path_parts = model(
                path_z, path_tau, masked_strain, masked_drug, bio, observation
            )
            integrated_total += path_velocity / config.quadrature_steps
            for index in range(4):
                integrated_parts[index] += path_parts[index] / config.quadrature_steps

        predicted_latent = z0 + integrated_total
        endpoint_loss = nn.functional.mse_loss(predicted_latent, z1)
        component_loss = sum(
            nn.functional.mse_loss(prediction, target)
            for prediction, target in zip(
                integrated_parts, (mean_t, drug_t, strain_t, interaction_t)
            )
        )
        flow_protein = predicted_latent @ pca_components + pca_mean
        residual = model.protein_residual(
            predicted_latent, masked_strain, masked_drug, bio, observation
        )
        protein_prediction = flow_protein + residual
        protein_squared = (protein_prediction - y_target).square() * protein_mask
        protein_loss = protein_squared.sum() / protein_mask.sum().clamp_min(1.0)

        mean_p, drug_p, strain_p, interaction_p, _, observation_p = parts
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
            + PROTEIN_ENDPOINT_WEIGHT * protein_loss
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals += np.array(
            [
                loss.item(), fm_loss.item(), endpoint_loss.item(),
                component_loss.item(), protein_loss.item(), orth_loss.item(),
            ]
        ) * len(z0)
        count += len(z0)
    values = totals / max(count, 1)
    names = ("total", "fm", "endpoint", "component", "protein", "orth")
    return dict(zip(names, values.tolist())), episode


def integrate_latent(
    model: ESIFMV4,
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
        tau = torch.full((len(z), 1), (step + 0.5) * dt)
        velocity, _ = model(z, tau, strain, drug, bio, observation)
        z = z + dt * velocity
    return z


def predict_rows(
    model: ESIFMV4,
    run: PreparedRun,
    rows: np.ndarray,
    config: V2Config,
    use_residual: bool,
) -> np.ndarray:
    strains, drugs, bio, observation = run.encoded
    pca_mean = torch.tensor(run.pca.mean_.astype(np.float32))
    pca_components = torch.tensor(run.pca.components_.astype(np.float32))
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch_rows = rows[start:start + 256]
            strain = torch.tensor(strains[batch_rows])
            drug = torch.tensor(drugs[batch_rows])
            batch_bio = torch.tensor(bio[batch_rows])
            batch_observation = torch.tensor(observation[batch_rows])
            latent = integrate_latent(
                model,
                torch.tensor(run.z0_all[batch_rows]),
                strain,
                drug,
                batch_bio,
                batch_observation,
                config.inference_steps,
            )
            protein = latent @ pca_components + pca_mean
            if use_residual:
                protein = protein + model.protein_residual(
                    latent, strain, drug, batch_bio, batch_observation
                )
            outputs.append(protein.numpy())
    return np.vstack(outputs)


def validation_rmse(
    model: ESIFMV4,
    run: PreparedRun,
    config: V2Config,
    use_residual: bool,
) -> float:
    prediction = predict_rows(model, run, run.validation_rows, config, use_residual)
    truth = run.y_log2.iloc[run.validation_rows].to_numpy()
    valid = np.isfinite(truth)
    squared = np.where(valid, (prediction - truth) ** 2, 0.0)
    return float(np.sqrt(squared.sum() / valid.sum()))


def select_epoch(
    run: PreparedRun,
    config: V2Config,
    episodic: bool,
) -> tuple[int, list[dict[str, object]]]:
    model = ESIFMV4(config, run.encoder, run.y_log2.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = build_loader(run, config)
    pca_mean = torch.tensor(run.pca.mean_.astype(np.float32))
    pca_components = torch.tensor(run.pca.components_.astype(np.float32))
    best_epoch = 1
    best_rmse = float("inf")
    stale = 0
    history = []
    for epoch in range(1, config.max_selection_epochs + 1):
        losses, episode = train_epoch(
            model, loader, optimizer, config, epoch, episodic, pca_mean, pca_components
        )
        rmse = validation_rmse(model, run, config, use_residual=True)
        flow_rmse = validation_rmse(model, run, config, use_residual=False)
        gate = float(torch.sigmoid(model.residual_gate_logit).item())
        record = {
            "epoch": epoch, **losses, "pseudo_s3_rmse": rmse,
            "pseudo_s3_flow_only_rmse": flow_rmse, "residual_gate": gate,
            "episode": episode,
        }
        history.append(record)
        if rmse < best_rmse - 1e-5:
            best_rmse = rmse
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"selection {epoch}: pseudo-S3={rmse:.5f}, flow={flow_rmse:.5f}, "
                f"best={best_rmse:.5f}, residual_gate={gate:.3f}"
            )
        if stale >= config.patience:
            break
    return best_epoch, history


def fit_full(
    run: PreparedRun,
    config: V2Config,
    epochs: int,
    episodic: bool,
) -> ESIFMV4:
    seed_everything(config.seed)
    model = ESIFMV4(config, run.encoder, run.y_log2.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = build_loader(run, config)
    pca_mean = torch.tensor(run.pca.mean_.astype(np.float32))
    pca_components = torch.tensor(run.pca.components_.astype(np.float32))
    for epoch in range(1, epochs + 1):
        losses, episode = train_epoch(
            model, loader, optimizer, config, epoch, episodic, pca_mean, pca_components
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            gate = float(torch.sigmoid(model.residual_gate_logit).item())
            print(
                f"full {epoch}/{epochs}: total={losses['total']:.5f}, "
                f"protein={losses['protein']:.5f}, gate={gate:.3f}, episode={episode}"
            )
    return model


def predict_all_versions(
    model: ESIFMV4,
    run: PreparedRun,
    config: V2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = np.arange(len(run.metadata))
    full = pd.DataFrame(
        predict_rows(model, run, rows, config, True),
        index=run.metadata.index,
        columns=run.y_log2.columns,
    )
    flow = pd.DataFrame(
        predict_rows(model, run, rows, config, False),
        index=run.metadata.index,
        columns=run.y_log2.columns,
    )
    control = run.metadata["perturbation_no_concentration"].isin(CONTROL_NAMES)
    control_values = run.control_prediction.loc[control].to_numpy(np.float32)
    full.loc[control] = control_values
    flow.loc[control] = control_values
    full.index.name = "sample_ID"
    flow.index.name = "sample_ID"
    return full, flow


def external_mlp(columns: pd.Index) -> pd.DataFrame:
    ids = np.load(
        f"{EXTERNAL_MLP_DIR}/condition_mlp_validation_sample_ids.npy", allow_pickle=False
    ).astype(str)
    values = np.load(
        f"{EXTERNAL_MLP_DIR}/condition_mlp_validation_predictions.npy", mmap_mode="r"
    )
    return pd.DataFrame(values, index=ids, columns=columns)


def main(episodic: bool) -> None:
    mode = "episodic" if episodic else "no_episodic"
    config = V2Config(
        latent_dim=128,
        hidden_dim=192,
        max_selection_epochs=45,
        patience=7,
        batch_size=192,
        quadrature_steps=3,
        inference_steps=8,
        seed=20260814,
    )
    seed_everything(config.seed)
    metadata = pd.read_csv(METADATA_PATH).set_index("sample_ID")
    proteome = pd.read_csv(PROTEOME_PATH).set_index("sample_ID")
    official_train = metadata["split_final"].eq("train").to_numpy()
    keep = proteome.iloc[np.flatnonzero(official_train)].isna().mean(axis=0) < 0.697
    y_log2 = np.log2(proteome.loc[:, keep])
    observation_mask = y_log2.notna().to_numpy()

    selection_fit, pseudo_validation, pseudo_info = pseudo_s3_masks(metadata)
    selection_run = prepare_run(
        metadata, y_log2, observation_mask, selection_fit, pseudo_validation, config
    )
    best_epoch, history = select_epoch(selection_run, config, episodic)
    print(f"selected full-training epochs: {best_epoch}")

    full_run = prepare_run(
        metadata, y_log2, observation_mask, official_train,
        np.zeros(len(metadata), dtype=bool), config,
    )
    model = fit_full(full_run, config, best_epoch, episodic)
    prediction, flow_only = predict_all_versions(model, full_run, config)
    output_path = OUTPUT_TEMPLATE.format(mode=mode)
    flow_output_path = FLOW_OUTPUT_TEMPLATE.format(mode=mode)
    comparison_path = COMPARISON_TEMPLATE.format(mode=mode)
    diagnostics_path = DIAGNOSTICS_TEMPLATE.format(mode=mode)
    prediction.to_csv(output_path)
    flow_only.to_csv(flow_output_path)

    baselines = {
        "Condition MLP": external_mlp(y_log2.columns),
        "ES-IFM-v2": pd.read_csv(
            "prediction_es_ifm_v2.csv", index_col="sample_ID",
            usecols=["sample_ID", *y_log2.columns],
        ),
    }
    rows = []
    for scenario, split in SCENARIOS.items():
        ids = metadata.index[
            metadata["split_final"].eq(split)
            & ~metadata["perturbation_no_concentration"].isin(EXCLUDED_TREATMENTS)
        ]
        truth = y_log2.loc[ids]
        candidates = [
            *baselines.items(),
            (f"ES-IFM-v4-{mode}-flow", flow_only),
            (f"ES-IFM-v4-{mode}", prediction),
        ]
        for name, values in candidates:
            rmse, global_r2, protein_r2 = masked_metrics(truth, values.loc[ids])
            rows.append({
                "场景": scenario, "模型": name, "样本数": len(ids),
                "log2 RMSE": rmse, "Global R²": global_r2,
                "蛋白 R² 中位数": protein_r2,
            })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    diagnostics = {
        "mode": mode,
        "config": asdict(config),
        "pseudo_s3": pseudo_info,
        "selected_epoch": best_epoch,
        "selection_history": history,
        "final_residual_gate": float(torch.sigmoid(model.residual_gate_logit).item()),
        "pca_explained_variance": float(full_run.pca.explained_variance_ratio_.sum()),
        "residual_limit": RESIDUAL_LIMIT,
        "protein_endpoint_weight": PROTEIN_ENDPOINT_WEIGHT,
    }
    Path(diagnostics_path).write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    printable = comparison.rename(
        columns={"Global R²": "Global R2", "蛋白 R² 中位数": "Protein R2 median"}
    )
    print(printable.to_string(index=False))
    print(f"saved: {output_path}")
    print(f"saved: {flow_output_path}")
    print(f"saved: {comparison_path}")
    print(f"saved: {diagnostics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-episodic", action="store_true")
    args = parser.parse_args()
    main(not args.no_episodic)
