from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


CONTROL_NAMES = {"Water", "DMSO"}
KEY_COLS = [
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
]
STRAIN_COL = "Strains"
DRUG_COL = "perturbation_no_concentration"
STRATEGIES = [
    "random",
    "diversity",
    "uncertainty",
    "interaction_uncertainty",
    "hybrid",
    "content_diversity",
    "content_hybrid",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_combo_responses(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(data_dir / "WAYB_WAYC_metadata_train_val(1).csv").set_index("sample_ID")
    abundance = pd.read_csv(data_dir / "WAYB_WAYC_proteome_raw_train_val.csv").set_index("sample_ID")
    abundance = abundance.reindex(metadata.index)
    train = metadata["split_final"].eq("train")
    controls = train & metadata[DRUG_COL].isin(CONTROL_NAMES)
    treatments = train & ~metadata[DRUG_COL].isin(CONTROL_NAMES)

    keep = abundance.loc[train].isna().mean(axis=0) < 0.697
    y_log2 = np.log2(abundance.loc[:, keep])
    key_strings = metadata[KEY_COLS].astype(str).agg("\x1f".join, axis=1)
    control_means = y_log2.loc[controls].groupby(key_strings.loc[controls]).mean()

    matched = treatments & key_strings.isin(control_means.index)
    treatment_meta = metadata.loc[matched].reset_index(drop=True)
    treatment_values = y_log2.loc[matched].to_numpy(dtype=np.float32)
    matched_controls = control_means.reindex(key_strings.loc[matched]).to_numpy(dtype=np.float32)
    deltas = treatment_values - matched_controls

    protein_cols = y_log2.columns.tolist()
    combo_keys = treatment_meta[[STRAIN_COL, DRUG_COL]].reset_index(drop=True)
    delta_frame = pd.concat(
        [combo_keys, pd.DataFrame(deltas, columns=protein_cols)], axis=1, copy=False
    )
    combo_response = delta_frame.groupby([STRAIN_COL, DRUG_COL], sort=True)[protein_cols].mean()
    combos = combo_response.index.to_frame(index=False)
    target = combo_response.to_numpy(dtype=np.float32)
    observed = np.isfinite(target)

    strains = sorted(combos[STRAIN_COL].unique())
    drugs = sorted(combos[DRUG_COL].unique())
    expected = len(strains) * len(drugs)
    if len(combos) != expected:
        raise ValueError(f"Expected a complete strain-drug grid ({expected}), found {len(combos)}")

    print(
        f"Loaded {len(combos)} combinations ({len(strains)} strains x {len(drugs)} drugs), "
        f"using {int(matched.sum())} strictly matched treatment samples."
    )
    return combos, target, observed


def load_content_representations(
    combos: pd.DataFrame,
    strain_feature_path: Path,
    chemical_feature_path: Path,
    seed: int,
) -> tuple[np.ndarray, dict]:
    strain_archive = np.load(strain_feature_path, allow_pickle=False)
    strain_names = strain_archive["entity_names"].astype(str)
    strain_features = np.clip(strain_archive["features"].astype(np.float32), -8.0, 8.0)
    strain_lookup = dict(zip(strain_names, strain_features))

    chemical_archive = np.load(chemical_feature_path, allow_pickle=False)
    chemical_names = chemical_archive["entity_names"].astype(str)
    chemical_features = chemical_archive["features"].astype(np.float32)
    feature_names = chemical_archive["feature_names"].astype(str)
    fingerprint_mask = np.array(
        [name.startswith("morgan_") or name.startswith("maccs_") for name in feature_names]
    )
    descriptor_mask = np.array([name.startswith("descriptor_z_") for name in feature_names])
    available_mask = feature_names == "molecule_feature_available"

    missing_strains = sorted(set(combos[STRAIN_COL].astype(str)) - set(strain_lookup))
    missing_chemicals = sorted(set(combos[DRUG_COL].astype(str)) - set(chemical_names))
    if missing_strains or missing_chemicals:
        raise ValueError(
            f"Missing content features: strains={missing_strains}, chemicals={missing_chemicals}"
        )

    # This compression is fitted only on molecular structure bits, never response labels.
    fingerprint_pca = PCA(n_components=16, svd_solver="randomized", random_state=seed)
    fingerprint_scores = fingerprint_pca.fit_transform(chemical_features[:, fingerprint_mask])
    fingerprint_scores /= np.sqrt(np.maximum(fingerprint_pca.explained_variance_, 1e-6))
    descriptor_values = chemical_features[:, descriptor_mask]
    availability = chemical_features[:, available_mask]
    chemical_content = np.concatenate(
        [fingerprint_scores, descriptor_values, availability], axis=1
    ).astype(np.float32)
    chemical_lookup = dict(zip(chemical_names, chemical_content))

    strain_rows = np.stack([strain_lookup[name] for name in combos[STRAIN_COL].astype(str)])
    chemical_rows = np.stack([chemical_lookup[name] for name in combos[DRUG_COL].astype(str)])
    strain_rows /= np.linalg.norm(strain_rows, axis=1, keepdims=True).clip(min=1e-6)
    chemical_rows /= np.linalg.norm(chemical_rows, axis=1, keepdims=True).clip(min=1e-6)
    pair_content = np.einsum("bi,bj->bij", strain_rows, chemical_rows).reshape(len(combos), -1)
    pair_content /= np.linalg.norm(pair_content, axis=1, keepdims=True).clip(min=1e-6)
    audit = {
        "strain_source": str(strain_feature_path),
        "strain_dimension": int(strain_features.shape[1]),
        "chemical_source": str(chemical_feature_path),
        "chemical_fingerprint_dimension": int(fingerprint_mask.sum()),
        "chemical_descriptor_dimension": int(descriptor_mask.sum()),
        "chemical_pca_dimension": 16,
        "pair_interaction_dimension": int(pair_content.shape[1]),
        "response_labels_used": False,
    }
    print(
        f"Loaded content features: strain={strain_features.shape[1]}D, "
        f"chemical={chemical_content.shape[1]}D, pair={pair_content.shape[1]}D."
    )
    return pair_content.astype(np.float32), audit


class InteractionModel(torch.nn.Module):
    def __init__(self, n_strains: int, n_drugs: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.global_bias = torch.nn.Parameter(torch.zeros(output_dim))
        self.strain_bias = torch.nn.Embedding(n_strains, output_dim)
        self.drug_bias = torch.nn.Embedding(n_drugs, output_dim)
        self.strain_factor = torch.nn.Embedding(n_strains, rank)
        self.drug_factor = torch.nn.Embedding(n_drugs, rank)
        self.interaction_basis = torch.nn.Parameter(torch.empty(rank, output_dim))
        torch.nn.init.zeros_(self.strain_bias.weight)
        torch.nn.init.zeros_(self.drug_bias.weight)
        torch.nn.init.normal_(self.strain_factor.weight, std=0.10)
        torch.nn.init.normal_(self.drug_factor.weight, std=0.10)
        torch.nn.init.normal_(self.interaction_basis, std=0.05)

    def forward(self, strain: torch.Tensor, drug: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        interaction = (self.strain_factor(strain) * self.drug_factor(drug)) @ self.interaction_basis
        prediction = (
            self.global_bias
            + self.strain_bias(strain)
            + self.drug_bias(drug)
            + interaction
        )
        return prediction, interaction


@dataclass
class EnsemblePrediction:
    decoded_mean: np.ndarray
    latent_members: np.ndarray
    interaction_members: np.ndarray


def fit_ensemble(
    target: np.ndarray,
    strain_ids: np.ndarray,
    drug_ids: np.ndarray,
    labelled: np.ndarray,
    ensemble_size: int,
    epochs: int,
    seed: int,
) -> EnsemblePrediction:
    labelled_target = target[labelled]
    protein_mean = np.nanmean(labelled_target, axis=0)
    protein_mean = np.nan_to_num(protein_mean, nan=0.0)
    filled = np.where(np.isfinite(labelled_target), labelled_target, protein_mean)

    n_components = min(24, len(labelled) - 1, filled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(filled)
    scales = np.sqrt(np.maximum(pca.explained_variance_, 1e-6))
    whitened = scores / scales

    all_strain = torch.as_tensor(strain_ids, dtype=torch.long)
    all_drug = torch.as_tensor(drug_ids, dtype=torch.long)
    train_strain = all_strain[labelled]
    train_drug = all_drug[labelled]
    train_target = torch.as_tensor(whitened, dtype=torch.float32)
    latent_members: list[np.ndarray] = []
    interaction_members: list[np.ndarray] = []

    for member in range(ensemble_size):
        member_seed = seed + 1009 * member
        seed_everything(member_seed)
        model = InteractionModel(
            n_strains=int(strain_ids.max()) + 1,
            n_drugs=int(drug_ids.max()) + 1,
            output_dim=n_components,
            rank=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=2e-3)
        rng = np.random.default_rng(member_seed)
        weights = torch.as_tensor(rng.exponential(1.0, len(labelled)), dtype=torch.float32)
        weights = weights / weights.mean()

        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(train_strain, train_drug)
            loss = ((prediction - train_target).square().mean(dim=1) * weights).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction, interaction = model(all_strain, all_drug)
        latent_members.append(prediction.numpy())
        interaction_members.append(interaction.numpy())

    latent_array = np.stack(latent_members)
    interaction_array = np.stack(interaction_members)
    decoded_mean = pca.inverse_transform(latent_array.mean(axis=0) * scales).astype(np.float32)
    return EnsemblePrediction(decoded_mean, latent_array, interaction_array)


def make_split(combos: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    strain_values = sorted(combos[STRAIN_COL].unique())
    drug_values = sorted(combos[DRUG_COL].unique())
    lookup = {(row[STRAIN_COL], row[DRUG_COL]): idx for idx, row in combos.iterrows()}

    for _ in range(1000):
        held_by_drug = {drug: rng.choice(strain_values) for drug in drug_values}
        if len(set(held_by_drug.values())) < len(strain_values):
            continue
        evaluation = np.array([lookup[(held_by_drug[drug], drug)] for drug in drug_values])
        initial = []
        for drug in drug_values:
            candidates = [strain for strain in strain_values if strain != held_by_drug[drug]]
            initial.append(lookup[(rng.choice(candidates), drug)])
        initial = np.array(initial)
        if len(set(combos.iloc[initial][STRAIN_COL])) == len(strain_values):
            pool = np.setdiff1d(np.arange(len(combos)), evaluation)
            return evaluation, initial, pool
    raise RuntimeError("Could not construct a balanced replay split")


def rowwise_correlation(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    correlations = []
    for truth, prediction, valid in zip(y_true, y_pred, mask):
        if valid.sum() < 2:
            continue
        a = truth[valid]
        b = prediction[valid]
        a = a - a.mean()
        b = b - b.mean()
        denominator = np.sqrt(np.square(a).sum() * np.square(b).sum())
        if denominator > 0:
            correlations.append(float(np.dot(a, b) / denominator))
    return float(np.mean(correlations)) if correlations else float("nan")


def evaluate(target: np.ndarray, observed: np.ndarray, prediction: np.ndarray, indices: np.ndarray) -> dict:
    valid = observed[indices]
    error = prediction[indices] - target[indices]
    rmse = float(np.sqrt(np.square(error[valid]).mean()))
    correlation = rowwise_correlation(target[indices], prediction[indices], valid)
    return {"rmse": rmse, "mean_row_correlation": correlation}


def normalize(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / (spread + 1e-8)


def select_batch(
    strategy: str,
    candidates: np.ndarray,
    labelled: np.ndarray,
    prediction: EnsemblePrediction,
    content_representation: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if strategy == "random":
        return rng.choice(candidates, size=batch_size, replace=False)

    latent_mean = prediction.latent_members.mean(axis=0)
    uncertainty = prediction.latent_members.var(axis=0).mean(axis=1)
    interaction_uncertainty = prediction.interaction_members.var(axis=0).mean(axis=1)
    selected: list[int] = []
    remaining = candidates.tolist()

    while len(selected) < batch_size:
        reference = np.concatenate([labelled, np.asarray(selected, dtype=int)])
        candidate_array = np.asarray(remaining, dtype=int)
        latent_distances = np.linalg.norm(
            latent_mean[candidate_array, None, :] - latent_mean[reference][None, :, :], axis=2
        ).min(axis=1)
        content_distances = np.linalg.norm(
            content_representation[candidate_array, None, :]
            - content_representation[reference][None, :, :],
            axis=2,
        ).min(axis=1)
        if strategy == "diversity":
            scores = latent_distances
        elif strategy == "uncertainty":
            scores = uncertainty[candidate_array]
        elif strategy == "interaction_uncertainty":
            scores = interaction_uncertainty[candidate_array]
        elif strategy == "hybrid":
            scores = normalize(uncertainty[candidate_array]) + normalize(latent_distances)
        elif strategy == "content_diversity":
            scores = content_distances
        elif strategy == "content_hybrid":
            scores = normalize(uncertainty[candidate_array]) + normalize(content_distances)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        chosen_position = int(np.argmax(scores))
        selected.append(remaining.pop(chosen_position))
    return np.asarray(selected, dtype=int)


def run_replay(
    combos: pd.DataFrame,
    target: np.ndarray,
    observed: np.ndarray,
    content_representation: np.ndarray,
    strategy: str,
    repeat: int,
    budgets: list[int],
    ensemble_size: int,
    epochs: int,
    base_seed: int,
) -> list[dict]:
    seed = base_seed + repeat * 10000
    evaluation, initial, pool = make_split(combos, seed)
    labelled = initial.copy()
    rng = np.random.default_rng(seed + STRATEGIES.index(strategy) * 101)
    strain_ids = pd.Categorical(combos[STRAIN_COL]).codes.copy()
    drug_ids = pd.Categorical(combos[DRUG_COL]).codes.copy()
    rows = []

    for round_index, budget in enumerate(budgets):
        if len(labelled) != budget:
            raise RuntimeError(f"Replay budget mismatch: {len(labelled)} != {budget}")
        fitted = fit_ensemble(
            target,
            strain_ids,
            drug_ids,
            labelled,
            ensemble_size,
            epochs,
            seed + round_index * 313,
        )
        metrics = evaluate(target, observed, fitted.decoded_mean, evaluation)
        rows.append(
            {
                "repeat": repeat,
                "strategy": strategy,
                "budget": budget,
                "evaluation_combinations": len(evaluation),
                **metrics,
            }
        )
        print(
            f"repeat={repeat + 1} strategy={strategy:25s} budget={budget:3d} "
            f"rmse={metrics['rmse']:.5f} corr={metrics['mean_row_correlation']:.4f}"
        )
        if round_index == len(budgets) - 1:
            continue
        next_budget = budgets[round_index + 1]
        candidates = np.setdiff1d(pool, labelled)
        acquired = select_batch(
            strategy,
            candidates,
            labelled,
            fitted,
            content_representation,
            next_budget - budget,
            rng,
        )
        labelled = np.concatenate([labelled, acquired])
    return rows


def summarize(results: pd.DataFrame, output_dir: Path, strategies: list[str]) -> None:
    summary = (
        results.groupby(["strategy", "budget"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            correlation_mean=("mean_row_correlation", "mean"),
            correlation_std=("mean_row_correlation", "std"),
        )
        .sort_values(["budget", "rmse_mean"])
    )
    summary.to_csv(output_dir / "active_learning_summary.csv", index=False)

    aulc_rows = []
    for (repeat, strategy), frame in results.groupby(["repeat", "strategy"]):
        frame = frame.sort_values("budget")
        area = float(np.trapezoid(frame["rmse"], frame["budget"]))
        width = float(frame["budget"].iloc[-1] - frame["budget"].iloc[0])
        aulc_rows.append({"repeat": repeat, "strategy": strategy, "normalized_aulc": area / width})
    aulc = pd.DataFrame(aulc_rows)
    random_aulc = aulc[aulc["strategy"].eq("random")][["repeat", "normalized_aulc"]].rename(
        columns={"normalized_aulc": "random_aulc"}
    )
    comparison = aulc.merge(random_aulc, on="repeat")
    comparison["improvement_vs_random"] = comparison["random_aulc"] - comparison["normalized_aulc"]
    comparison.to_csv(output_dir / "active_learning_aulc.csv", index=False)

    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for strategy in strategies:
        frame = summary[summary["strategy"].eq(strategy)].sort_values("budget")
        axis.errorbar(
            frame["budget"],
            frame["rmse_mean"],
            yerr=frame["rmse_std"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=strategy.replace("_", " "),
        )
    axis.set_xlabel("Revealed strain-drug combinations")
    axis.set_ylabel("Held-combination RMSE (log2 response)")
    axis.set_title("Offline active-learning replay")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "active_learning_curve.png", dpi=180)
    plt.close(fig)

    final_budget = int(results["budget"].max())
    report = {
        "protocol": {
            "target": "mean strict-control-adjusted log2 proteomic response per strain-drug combination",
            "evaluation": "one permanently hidden strain combination per drug",
            "initial_design": "one revealed non-evaluation strain combination per drug",
            "selection_pool_excludes_evaluation": True,
            "hidden_labels_used_for_selection": False,
        },
        "final_budget": final_budget,
        "final_ranking": summary[summary["budget"].eq(final_budget)][
            ["strategy", "rmse_mean", "rmse_std", "correlation_mean"]
        ].to_dict(orient="records"),
        "aulc_improvement_vs_random": (
            comparison.groupby("strategy")["improvement_vs_random"].agg(["mean", "std"]).reset_index().to_dict(orient="records")
        ),
    }
    (output_dir / "active_learning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nFinal-budget ranking:")
    print(summary[summary["budget"].eq(final_budget)].to_string(index=False))
    print("\nAULC improvement versus random (positive is better):")
    print(
        comparison.groupby("strategy")["improvement_vs_random"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .to_string()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free offline active-learning replay")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("active_learning_run"))
    parser.add_argument(
        "--strain-features",
        type=Path,
        default=Path("external_strain_features/strain_sequence_embeddings.npz"),
    )
    parser.add_argument(
        "--chemical-features",
        type=Path,
        default=Path("external_strain_features/chemical_features.npz"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="Comma-separated acquisition strategies",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combos, target, observed = load_combo_responses(args.data_dir)
    content_representation, content_audit = load_content_representations(
        combos, args.strain_features, args.chemical_features, args.seed
    )
    strategies = [strategy.strip() for strategy in args.strategies.split(",") if strategy.strip()]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown or "random" not in strategies:
        raise ValueError(f"Strategies must include random and contain no unknown values: {unknown}")
    initial_budget = combos[DRUG_COL].nunique()
    budgets = [initial_budget + 16 * step for step in range(5)]
    rows = []
    for repeat in range(args.repeats):
        for strategy in strategies:
            rows.extend(
                run_replay(
                    combos,
                    target,
                    observed,
                    content_representation,
                    strategy,
                    repeat,
                    budgets,
                    args.ensemble_size,
                    args.epochs,
                    args.seed,
                )
            )
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "active_learning_results.csv", index=False)
    summarize(results, args.output_dir, strategies)
    (args.output_dir / "content_feature_audit.json").write_text(
        json.dumps(content_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
