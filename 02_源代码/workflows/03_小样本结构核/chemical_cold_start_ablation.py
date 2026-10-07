from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from active_learning_replay import DRUG_COL, STRAIN_COL, evaluate, load_combo_responses, seed_everything
from content_predictor_ablation import load_entity_content


VARIANTS = ["id_oov", "content_only", "content_residual"]


class ColdStartPredictor(torch.nn.Module):
    def __init__(
        self,
        strain_dim: int,
        chemical_dim: int,
        output_dim: int,
        n_strains: int,
        n_known_drugs: int,
        variant: str,
        rank: int = 4,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.global_bias = torch.nn.Parameter(torch.zeros(output_dim))

        if variant == "id_oov":
            self.strain_id_main = torch.nn.Embedding(n_strains, output_dim)
            self.drug_id_main = torch.nn.Embedding(n_known_drugs + 1, output_dim, padding_idx=0)
            self.strain_id_factor = torch.nn.Embedding(n_strains, rank)
            self.drug_id_factor = torch.nn.Embedding(n_known_drugs + 1, rank, padding_idx=0)
            self.id_basis = torch.nn.Parameter(torch.empty(rank, output_dim))
            torch.nn.init.zeros_(self.strain_id_main.weight)
            torch.nn.init.zeros_(self.drug_id_main.weight)
            torch.nn.init.normal_(self.strain_id_factor.weight, std=0.05)
            torch.nn.init.normal_(self.drug_id_factor.weight, std=0.05)
            torch.nn.init.normal_(self.id_basis, std=0.03)
            return

        self.strain_content_main = torch.nn.Linear(strain_dim, output_dim, bias=False)
        self.drug_content_main = torch.nn.Linear(chemical_dim, output_dim, bias=False)
        self.strain_content_factor = torch.nn.Linear(strain_dim, rank)
        self.drug_content_factor = torch.nn.Linear(chemical_dim, rank)
        self.content_basis = torch.nn.Parameter(torch.empty(rank, output_dim))
        torch.nn.init.normal_(self.content_basis, std=0.04)

        if variant == "content_residual":
            self.strain_id_main = torch.nn.Embedding(n_strains, output_dim)
            self.drug_id_main = torch.nn.Embedding(n_known_drugs + 1, output_dim, padding_idx=0)
            self.strain_id_factor = torch.nn.Embedding(n_strains, rank)
            self.drug_id_factor = torch.nn.Embedding(n_known_drugs + 1, rank, padding_idx=0)
            self.id_basis = torch.nn.Parameter(torch.empty(rank, output_dim))
            torch.nn.init.zeros_(self.strain_id_main.weight)
            torch.nn.init.zeros_(self.drug_id_main.weight)
            torch.nn.init.normal_(self.strain_id_factor.weight, std=0.04)
            torch.nn.init.normal_(self.drug_id_factor.weight, std=0.04)
            torch.nn.init.normal_(self.id_basis, std=0.02)

    def id_prediction(
        self, strain_id: torch.Tensor, drug_id: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.strain_id_main(strain_id)
            + self.drug_id_main(drug_id)
            + (self.strain_id_factor(strain_id) * self.drug_id_factor(drug_id)) @ self.id_basis
        )

    def forward(
        self,
        strain_content: torch.Tensor,
        chemical_content: torch.Tensor,
        strain_id: torch.Tensor,
        drug_id: torch.Tensor,
        residual_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "id_oov":
            residual = self.id_prediction(strain_id, drug_id)
            return self.global_bias + residual, residual

        content_interaction = (
            torch.tanh(self.strain_content_factor(strain_content))
            * torch.tanh(self.drug_content_factor(chemical_content))
        ) @ self.content_basis
        prediction = (
            self.global_bias
            + self.strain_content_main(strain_content)
            + self.drug_content_main(chemical_content)
            + content_interaction
        )
        residual = torch.zeros_like(prediction)
        if self.variant == "content_residual":
            residual = self.id_prediction(strain_id, drug_id)
            if residual_mask is not None:
                residual = residual * residual_mask
            prediction = prediction + residual
        return prediction, residual


def make_drug_split(
    combos: pd.DataFrame, repeat: int, seed: int, n_evaluation: int
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed + repeat * 10000)
    drugs = np.array(sorted(combos[DRUG_COL].astype(str).unique()))
    evaluation_drugs = rng.choice(drugs, size=n_evaluation, replace=False)
    development_drugs = np.setdiff1d(drugs, evaluation_drugs)
    return evaluation_drugs, rng.permutation(development_drugs).tolist()


def fit_predictor(
    variant: str,
    target: np.ndarray,
    labelled: np.ndarray,
    known_drugs: list[str],
    combos: pd.DataFrame,
    strain_content: np.ndarray,
    chemical_content: np.ndarray,
    ensemble_size: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    labelled_target = target[labelled]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        protein_mean = np.nanmean(labelled_target, axis=0)
    protein_mean = np.nan_to_num(protein_mean, nan=0.0)
    filled = np.where(np.isfinite(labelled_target), labelled_target, protein_mean)
    n_components = min(32, len(labelled) - 1, filled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    train_scores = pca.fit_transform(filled).astype(np.float32)

    strain_ids = pd.Categorical(combos[STRAIN_COL]).codes.copy()
    drug_lookup = {name: index + 1 for index, name in enumerate(known_drugs)}
    drug_ids = combos[DRUG_COL].astype(str).map(drug_lookup).fillna(0).astype(int).to_numpy()
    all_strain_ids = torch.tensor(strain_ids, dtype=torch.long)
    all_drug_ids = torch.tensor(drug_ids, dtype=torch.long)
    all_strain_content = torch.tensor(strain_content, dtype=torch.float32)
    all_chemical_content = torch.tensor(chemical_content, dtype=torch.float32)
    train_target = torch.tensor(train_scores, dtype=torch.float32)
    members = []

    for member in range(ensemble_size):
        member_seed = seed + member * 1009
        seed_everything(member_seed)
        model = ColdStartPredictor(
            strain_content.shape[1], chemical_content.shape[1], n_components,
            int(strain_ids.max()) + 1, len(known_drugs), variant,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=4e-3)
        rng = np.random.default_rng(member_seed)
        weights = torch.tensor(rng.exponential(1.0, len(labelled)), dtype=torch.float32)
        weights /= weights.mean()

        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            residual_mask = None
            if variant == "content_residual":
                residual_mask = (torch.rand(len(labelled), 1) >= 0.35).float()
            prediction, residual = model(
                all_strain_content[labelled], all_chemical_content[labelled],
                all_strain_ids[labelled], all_drug_ids[labelled], residual_mask,
            )
            per_row = (prediction - train_target).square().mean(dim=1)
            loss = (per_row * weights).mean()
            if variant == "content_residual":
                loss = loss + 0.10 * residual.square().mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction, _ = model(
                all_strain_content, all_chemical_content, all_strain_ids, all_drug_ids
            )
        members.append(prediction.numpy())

    return pca.inverse_transform(np.stack(members).mean(axis=0)).astype(np.float32)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(20000)])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize(results: pd.DataFrame, output_dir: Path, seed: int) -> None:
    summary = (
        results.groupby(["variant", "known_drugs"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            correlation_mean=("mean_row_correlation", "mean"),
            correlation_std=("mean_row_correlation", "std"),
        )
    )
    summary.to_csv(output_dir / "chemical_cold_start_summary.csv", index=False)
    wide = results.pivot(index=["repeat", "known_drugs"], columns="variant", values="rmse")
    rows = []
    for budget in sorted(results["known_drugs"].unique()):
        frame = wide.xs(budget, level="known_drugs")
        for variant in ["content_only", "content_residual"]:
            delta = (frame["id_oov"] - frame[variant]).to_numpy()
            low, high = bootstrap_ci(delta, seed + budget + VARIANTS.index(variant))
            rows.append({
                "known_drugs": budget, "variant": variant,
                "rmse_improvement_vs_id_oov": float(delta.mean()),
                "ci95_low": low, "ci95_high": high,
                "wins": int((delta > 0).sum()), "repeats": len(delta),
            })
    paired = pd.DataFrame(rows)
    paired.to_csv(output_dir / "chemical_cold_start_paired.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for variant in VARIANTS:
        frame = summary[summary["variant"].eq(variant)].sort_values("known_drugs")
        axis.errorbar(
            frame["known_drugs"], frame["rmse_mean"], yerr=frame["rmse_std"],
            marker="o", capsize=3, label=variant.replace("_", " "),
        )
    axis.set_xlabel("Revealed chemicals (all four strains)")
    axis.set_ylabel("Fully unseen-chemical RMSE (log2 response)")
    axis.set_title("Chemical cold-start predictor ablation")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "chemical_cold_start_curve.png", dpi=180)
    plt.close(fig)
    print("\nMean metrics:")
    print(summary.to_string(index=False))
    print("\nPaired RMSE improvement versus ID-OOV (positive is better):")
    print(paired.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fully unseen-chemical response ablation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("chemical_cold_start_run"))
    parser.add_argument(
        "--strain-features", type=Path,
        default=Path("external_strain_features/strain_sequence_embeddings.npz"),
    )
    parser.add_argument(
        "--chemical-features", type=Path,
        default=Path("external_strain_features/chemical_features.npz"),
    )
    parser.add_argument("--evaluation-drugs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
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
    budgets = [8, 12, 16, 20, 24, 28]
    rows = []
    for repeat in range(args.repeats):
        evaluation_drugs, development_order = make_drug_split(
            combos, repeat, args.seed, args.evaluation_drugs
        )
        evaluation = np.flatnonzero(combos[DRUG_COL].astype(str).isin(evaluation_drugs).to_numpy())
        for budget in budgets:
            known_drugs = development_order[:budget]
            labelled = np.flatnonzero(combos[DRUG_COL].astype(str).isin(known_drugs).to_numpy())
            for variant in VARIANTS:
                prediction = fit_predictor(
                    variant, target, labelled, known_drugs, combos,
                    strain_content, chemical_content,
                    args.ensemble_size, args.epochs,
                    args.seed + repeat * 10000 + budget * 313,
                )
                metrics = evaluate(target, observed, prediction, evaluation)
                rows.append({
                    "repeat": repeat, "known_drugs": budget,
                    "evaluation_drugs": args.evaluation_drugs, "variant": variant, **metrics,
                })
                print(
                    f"repeat={repeat + 1:02d} drugs={budget:02d} variant={variant:16s} "
                    f"rmse={metrics['rmse']:.5f} corr={metrics['mean_row_correlation']:.4f}"
                )
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "chemical_cold_start_results.csv", index=False)
    summarize(results, args.output_dir, args.seed)
    audit.update({
        "protocol": "fully unseen evaluation chemicals",
        "evaluation_chemicals_per_repeat": args.evaluation_drugs,
        "id_for_unseen_chemical": "zero OOV vector",
        "loss_target": "unwhitened PCA response scores",
    })
    (args.output_dir / "chemical_cold_start_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
