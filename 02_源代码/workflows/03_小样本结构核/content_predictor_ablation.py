from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from active_learning_replay import (
    DRUG_COL,
    STRAIN_COL,
    InteractionModel,
    evaluate,
    load_combo_responses,
    make_split,
    seed_everything,
)


VARIANTS = ["id_only", "content_only", "content_residual"]


def load_entity_content(
    combos: pd.DataFrame,
    strain_path: Path,
    chemical_path: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    strain_archive = np.load(strain_path, allow_pickle=False)
    strain_names = strain_archive["entity_names"].astype(str)
    strain_values = np.clip(strain_archive["features"].astype(np.float32), -8.0, 8.0)
    strain_lookup = dict(zip(strain_names, strain_values))

    chemical_archive = np.load(chemical_path, allow_pickle=False)
    chemical_names = chemical_archive["entity_names"].astype(str)
    chemical_values = chemical_archive["features"].astype(np.float32)
    feature_names = chemical_archive["feature_names"].astype(str)
    fingerprints = np.array(
        [name.startswith("morgan_") or name.startswith("maccs_") for name in feature_names]
    )
    descriptors = np.array([name.startswith("descriptor_z_") for name in feature_names])
    availability = feature_names == "molecule_feature_available"

    missing_strains = sorted(set(combos[STRAIN_COL].astype(str)) - set(strain_names))
    missing_drugs = sorted(set(combos[DRUG_COL].astype(str)) - set(chemical_names))
    if missing_strains or missing_drugs:
        raise ValueError(f"Missing content: strains={missing_strains}, drugs={missing_drugs}")

    chemical_pca = PCA(n_components=16, svd_solver="randomized", random_state=seed)
    chemical_scores = chemical_pca.fit_transform(chemical_values[:, fingerprints])
    chemical_scores /= np.sqrt(np.maximum(chemical_pca.explained_variance_, 1e-6))
    chemical_content = np.concatenate(
        [chemical_scores, chemical_values[:, descriptors], chemical_values[:, availability]], axis=1
    ).astype(np.float32)
    chemical_lookup = dict(zip(chemical_names, chemical_content))

    strain_rows = np.stack([strain_lookup[name] for name in combos[STRAIN_COL].astype(str)])
    chemical_rows = np.stack([chemical_lookup[name] for name in combos[DRUG_COL].astype(str)])
    audit = {
        "strain_dimension": int(strain_rows.shape[1]),
        "chemical_dimension": int(chemical_rows.shape[1]),
        "chemical_fingerprint_bits": int(fingerprints.sum()),
        "chemical_descriptor_count": int(descriptors.sum()),
        "content_fit_uses_response_labels": False,
    }
    return strain_rows.astype(np.float32), chemical_rows.astype(np.float32), audit


class ContentPredictor(torch.nn.Module):
    def __init__(
        self,
        strain_dim: int,
        chemical_dim: int,
        output_dim: int,
        n_strains: int,
        n_drugs: int,
        use_residual: bool,
        rank: int = 4,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.global_bias = torch.nn.Parameter(torch.zeros(output_dim))
        self.strain_main = torch.nn.Linear(strain_dim, output_dim, bias=False)
        self.drug_main = torch.nn.Linear(chemical_dim, output_dim, bias=False)
        self.strain_factor = torch.nn.Linear(strain_dim, rank)
        self.drug_factor = torch.nn.Linear(chemical_dim, rank)
        self.content_basis = torch.nn.Parameter(torch.empty(rank, output_dim))
        torch.nn.init.normal_(self.content_basis, std=0.05)

        if use_residual:
            self.strain_residual = torch.nn.Embedding(n_strains, output_dim)
            self.drug_residual = torch.nn.Embedding(n_drugs, output_dim)
            self.strain_id_factor = torch.nn.Embedding(n_strains, rank)
            self.drug_id_factor = torch.nn.Embedding(n_drugs, rank)
            self.id_basis = torch.nn.Parameter(torch.empty(rank, output_dim))
            torch.nn.init.zeros_(self.strain_residual.weight)
            torch.nn.init.zeros_(self.drug_residual.weight)
            torch.nn.init.normal_(self.strain_id_factor.weight, std=0.05)
            torch.nn.init.normal_(self.drug_id_factor.weight, std=0.05)
            torch.nn.init.normal_(self.id_basis, std=0.02)

    def forward(
        self,
        strain_content: torch.Tensor,
        chemical_content: torch.Tensor,
        strain_id: torch.Tensor,
        drug_id: torch.Tensor,
        residual_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content_interaction = (
            torch.tanh(self.strain_factor(strain_content))
            * torch.tanh(self.drug_factor(chemical_content))
        ) @ self.content_basis
        prediction = (
            self.global_bias
            + self.strain_main(strain_content)
            + self.drug_main(chemical_content)
            + content_interaction
        )
        residual = torch.zeros_like(prediction)
        if self.use_residual:
            residual = (
                self.strain_residual(strain_id)
                + self.drug_residual(drug_id)
                + (self.strain_id_factor(strain_id) * self.drug_id_factor(drug_id)) @ self.id_basis
            )
            if residual_mask is not None:
                residual = residual * residual_mask
            prediction = prediction + residual
        return prediction, residual


def prepare_latent_targets(
    target: np.ndarray, labelled: np.ndarray, seed: int
) -> tuple[PCA, np.ndarray, np.ndarray]:
    labelled_target = target[labelled]
    protein_mean = np.nanmean(labelled_target, axis=0)
    protein_mean = np.nan_to_num(protein_mean, nan=0.0)
    filled = np.where(np.isfinite(labelled_target), labelled_target, protein_mean)
    n_components = min(24, len(labelled) - 1, filled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(filled)
    scales = np.sqrt(np.maximum(pca.explained_variance_, 1e-6))
    return pca, (scores / scales).astype(np.float32), scales.astype(np.float32)


def fit_variant(
    variant: str,
    target: np.ndarray,
    labelled: np.ndarray,
    strain_ids: np.ndarray,
    drug_ids: np.ndarray,
    strain_content: np.ndarray,
    chemical_content: np.ndarray,
    ensemble_size: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    pca, train_target_np, scales = prepare_latent_targets(target, labelled, seed)
    output_dim = train_target_np.shape[1]
    all_strain_ids = torch.tensor(strain_ids, dtype=torch.long)
    all_drug_ids = torch.tensor(drug_ids, dtype=torch.long)
    all_strain_content = torch.tensor(strain_content, dtype=torch.float32)
    all_chemical_content = torch.tensor(chemical_content, dtype=torch.float32)
    train_target = torch.tensor(train_target_np, dtype=torch.float32)
    members = []

    for member in range(ensemble_size):
        member_seed = seed + member * 1009
        seed_everything(member_seed)
        if variant == "id_only":
            model: torch.nn.Module = InteractionModel(
                int(strain_ids.max()) + 1, int(drug_ids.max()) + 1, output_dim, rank=4
            )
        else:
            model = ContentPredictor(
                strain_content.shape[1],
                chemical_content.shape[1],
                output_dim,
                int(strain_ids.max()) + 1,
                int(drug_ids.max()) + 1,
                use_residual=variant == "content_residual",
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=3e-3)
        rng = np.random.default_rng(member_seed)
        weights = torch.tensor(rng.exponential(1.0, len(labelled)), dtype=torch.float32)
        weights /= weights.mean()

        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            if variant == "id_only":
                prediction, _ = model(all_strain_ids[labelled], all_drug_ids[labelled])
                residual_penalty = torch.tensor(0.0)
            else:
                residual_mask = None
                if variant == "content_residual":
                    residual_mask = (torch.rand(len(labelled), 1) >= 0.35).float()
                prediction, residual = model(
                    all_strain_content[labelled],
                    all_chemical_content[labelled],
                    all_strain_ids[labelled],
                    all_drug_ids[labelled],
                    residual_mask,
                )
                residual_penalty = residual.square().mean()
            data_loss = ((prediction - train_target).square().mean(dim=1) * weights).mean()
            loss = data_loss + (0.08 * residual_penalty if variant == "content_residual" else 0.0)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if variant == "id_only":
                prediction, _ = model(all_strain_ids, all_drug_ids)
            else:
                prediction, _ = model(
                    all_strain_content,
                    all_chemical_content,
                    all_strain_ids,
                    all_drug_ids,
                )
        members.append(prediction.numpy())

    latent_mean = np.stack(members).mean(axis=0)
    return pca.inverse_transform(latent_mean * scales).astype(np.float32)


def labelled_paths(combos: pd.DataFrame, budgets: list[int], repeat: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    evaluation, initial, pool = make_split(combos, seed + repeat * 10000)
    rng = np.random.default_rng(seed + repeat * 10000 + 777)
    order = rng.permutation(np.setdiff1d(pool, initial))
    paths = [np.concatenate([initial, order[: budget - len(initial)]]) for budget in budgets]
    return evaluation, paths


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(20000)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(results: pd.DataFrame, output_dir: Path, seed: int) -> None:
    summary = (
        results.groupby(["variant", "budget"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            correlation_mean=("mean_row_correlation", "mean"),
            correlation_std=("mean_row_correlation", "std"),
        )
    )
    summary.to_csv(output_dir / "content_predictor_summary.csv", index=False)

    paired_rows = []
    wide = results.pivot(index=["repeat", "budget"], columns="variant", values="rmse")
    for budget in sorted(results["budget"].unique()):
        frame = wide.xs(budget, level="budget")
        for variant in ["content_only", "content_residual"]:
            delta = (frame["id_only"] - frame[variant]).to_numpy()
            low, high = bootstrap_ci(delta, seed + budget + VARIANTS.index(variant))
            paired_rows.append(
                {
                    "budget": budget,
                    "variant": variant,
                    "rmse_improvement_vs_id_mean": float(delta.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": int((delta > 0).sum()),
                    "repeats": len(delta),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "content_predictor_paired.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for variant in VARIANTS:
        frame = summary[summary["variant"].eq(variant)].sort_values("budget")
        axis.errorbar(
            frame["budget"], frame["rmse_mean"], yerr=frame["rmse_std"],
            marker="o", capsize=3, label=variant.replace("_", " ")
        )
    axis.set_xlabel("Revealed strain-drug combinations")
    axis.set_ylabel("Held-combination RMSE (log2 response)")
    axis.set_title("Content-conditioned predictor ablation")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "content_predictor_curve.png", dpi=180)
    plt.close(fig)

    print("\nMean metrics:")
    print(summary.to_string(index=False))
    print("\nPaired RMSE improvement versus ID-only (positive is better):")
    print(paired.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Content-conditioned response predictor ablation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("content_predictor_run"))
    parser.add_argument(
        "--strain-features", type=Path,
        default=Path("external_strain_features/strain_sequence_embeddings.npz"),
    )
    parser.add_argument(
        "--chemical-features", type=Path,
        default=Path("external_strain_features/chemical_features.npz"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combos, target, observed = load_combo_responses(args.data_dir)
    strain_content, chemical_content, audit = load_entity_content(
        combos, args.strain_features, args.chemical_features, args.seed
    )
    strain_ids = pd.Categorical(combos[STRAIN_COL]).codes.copy()
    drug_ids = pd.Categorical(combos[DRUG_COL]).codes.copy()
    initial_budget = combos[DRUG_COL].nunique()
    budgets = [initial_budget + 16 * step for step in range(5)]
    rows = []

    for repeat in range(args.repeats):
        evaluation, paths = labelled_paths(combos, budgets, repeat, args.seed)
        for budget, labelled in zip(budgets, paths):
            for variant in VARIANTS:
                prediction = fit_variant(
                    variant, target, labelled, strain_ids, drug_ids,
                    strain_content, chemical_content,
                    args.ensemble_size, args.epochs,
                    args.seed + repeat * 10000 + budgets.index(budget) * 313,
                )
                metrics = evaluate(target, observed, prediction, evaluation)
                rows.append({"repeat": repeat, "budget": budget, "variant": variant, **metrics})
                print(
                    f"repeat={repeat + 1:02d} budget={budget:3d} variant={variant:16s} "
                    f"rmse={metrics['rmse']:.5f} corr={metrics['mean_row_correlation']:.4f}"
                )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "content_predictor_results.csv", index=False)
    summarize(results, args.output_dir, args.seed)
    (args.output_dir / "content_predictor_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
