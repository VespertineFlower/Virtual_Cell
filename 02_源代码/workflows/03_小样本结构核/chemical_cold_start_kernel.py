from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from active_learning_replay import DRUG_COL, STRAIN_COL, evaluate, load_combo_responses
from chemical_cold_start_ablation import bootstrap_ci, fit_predictor, make_drug_split
from content_predictor_ablation import load_entity_content


def rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    distance = (
        np.square(left).sum(axis=1, keepdims=True)
        + np.square(right).sum(axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.exp(-gamma * np.maximum(distance, 0.0))


def solve_kernel(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    kernel = rbf_kernel(train_x, train_x, gamma)
    coefficients = np.linalg.solve(
        kernel + alpha * np.eye(len(train_x), dtype=np.float32), train_y
    )
    return rbf_kernel(query_x, train_x, gamma) @ coefficients


def select_hyperparameters(
    features: np.ndarray, targets: np.ndarray
) -> tuple[float, float, float]:
    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    gammas = [0.002, 0.005, 0.01, 0.02, 0.05]
    best = (float("inf"), alphas[0], gammas[0])
    for alpha in alphas:
        for gamma in gammas:
            errors = []
            for held in range(len(features)):
                fit = np.arange(len(features)) != held
                prediction = solve_kernel(
                    features[fit], targets[fit], features[held : held + 1], alpha, gamma
                )[0]
                errors.append(np.square(prediction - targets[held]).mean())
            error = float(np.mean(errors))
            if error < best[0]:
                best = (error, alpha, gamma)
    return best[1], best[2], best[0]


def fit_kernel_content(
    target: np.ndarray,
    labelled: np.ndarray,
    known_drugs: list[str],
    combos: pd.DataFrame,
    chemical_content: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    labelled_target = target[labelled]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        protein_mean = np.nanmean(labelled_target, axis=0)
    protein_mean = np.nan_to_num(protein_mean, nan=0.0)
    filled = np.where(np.isfinite(labelled_target), labelled_target, protein_mean)
    n_components = min(32, len(labelled) - 1, filled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    labelled_scores = pca.fit_transform(filled).astype(np.float32)

    labelled_meta = combos.iloc[labelled].copy()
    labelled_meta["_row"] = np.arange(len(labelled))
    score_frame = pd.DataFrame(labelled_scores)
    score_frame[DRUG_COL] = labelled_meta[DRUG_COL].astype(str).to_numpy()
    drug_targets = score_frame.groupby(DRUG_COL, sort=False).mean()
    known_order = drug_targets.index.astype(str).tolist()

    chemical_lookup = {
        name: chemical_content[np.flatnonzero(combos[DRUG_COL].astype(str).eq(name).to_numpy())[0]]
        for name in combos[DRUG_COL].astype(str).unique()
    }
    train_features = np.stack([chemical_lookup[name] for name in known_order])
    all_drugs = combos[DRUG_COL].astype(str).unique().tolist()
    all_drug_features = np.stack([chemical_lookup[name] for name in all_drugs])
    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features).astype(np.float32)
    all_drug_features = scaler.transform(all_drug_features).astype(np.float32)

    alpha, gamma, cv_mse = select_hyperparameters(train_features, drug_targets.to_numpy())
    predicted_drugs = solve_kernel(
        train_features,
        drug_targets.to_numpy(dtype=np.float32),
        all_drug_features,
        alpha,
        gamma,
    )
    drug_prediction = dict(zip(all_drugs, predicted_drugs))

    fitted_known = solve_kernel(
        train_features,
        drug_targets.to_numpy(dtype=np.float32),
        train_features,
        alpha,
        gamma,
    )
    known_fitted = dict(zip(known_order, fitted_known))
    residuals = labelled_scores - np.stack(
        [known_fitted[name] for name in labelled_meta[DRUG_COL].astype(str)]
    )
    strain_offsets = {}
    for strain in combos[STRAIN_COL].astype(str).unique():
        rows = labelled_meta[STRAIN_COL].astype(str).eq(strain).to_numpy()
        strain_offsets[strain] = residuals[rows].mean(axis=0)

    prediction_scores = np.stack(
        [
            drug_prediction[drug] + strain_offsets[strain]
            for strain, drug in zip(
                combos[STRAIN_COL].astype(str), combos[DRUG_COL].astype(str)
            )
        ]
    )
    prediction = pca.inverse_transform(prediction_scores).astype(np.float32)
    return prediction, {
        "alpha": alpha,
        "gamma": gamma,
        "leave_one_drug_out_score_mse": cv_mse,
    }


def summarize(results: pd.DataFrame, output_dir: Path, seed: int) -> None:
    summary = (
        results.groupby(["variant", "known_drugs"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            correlation_mean=("mean_row_correlation", "mean"),
            correlation_std=("mean_row_correlation", "std"),
        )
    )
    summary.to_csv(output_dir / "kernel_cold_start_summary.csv", index=False)
    wide = results.pivot(index=["repeat", "known_drugs"], columns="variant", values="rmse")
    paired_rows = []
    for budget in sorted(results["known_drugs"].unique()):
        frame = wide.xs(budget, level="known_drugs")
        delta = (frame["id_oov"] - frame["kernel_content"]).to_numpy()
        low, high = bootstrap_ci(delta, seed + budget)
        paired_rows.append({
            "known_drugs": budget,
            "rmse_improvement_vs_id_oov": float(delta.mean()),
            "ci95_low": low, "ci95_high": high,
            "wins": int((delta > 0).sum()), "repeats": len(delta),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "kernel_cold_start_paired.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.4, 4.7))
    for variant in ["id_oov", "kernel_content"]:
        frame = summary[summary["variant"].eq(variant)].sort_values("known_drugs")
        axis.errorbar(
            frame["known_drugs"], frame["rmse_mean"], yerr=frame["rmse_std"],
            marker="o", capsize=3, label=variant.replace("_", " "),
        )
    axis.set_xlabel("Revealed chemicals (all four strains)")
    axis.set_ylabel("Fully unseen-chemical RMSE (log2 response)")
    axis.set_title("Kernel content model for chemical cold start")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_cold_start_curve.png", dpi=180)
    plt.close(fig)
    print("\nMean metrics:")
    print(summary.to_string(index=False))
    print("\nPaired RMSE improvement versus ID-OOV (positive is better):")
    print(paired.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-sample kernel chemical cold-start model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("chemical_cold_start_kernel_run"))
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
    parser.add_argument("--id-epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combos, target, observed = load_combo_responses(args.data_dir)
    strain_content, chemical_content, audit = load_entity_content(
        combos, args.strain_features, args.chemical_features, args.seed
    )
    budgets = [8, 12, 16, 20, 24, 28]
    rows = []
    hyperparameters = []
    for repeat in range(args.repeats):
        evaluation_drugs, development_order = make_drug_split(
            combos, repeat, args.seed, args.evaluation_drugs
        )
        evaluation = np.flatnonzero(combos[DRUG_COL].astype(str).isin(evaluation_drugs).to_numpy())
        for budget in budgets:
            known_drugs = development_order[:budget]
            labelled = np.flatnonzero(combos[DRUG_COL].astype(str).isin(known_drugs).to_numpy())
            id_prediction = fit_predictor(
                "id_oov", target, labelled, known_drugs, combos,
                strain_content, chemical_content, 3, args.id_epochs,
                args.seed + repeat * 10000 + budget * 313,
            )
            kernel_prediction, selected = fit_kernel_content(
                target, labelled, known_drugs, combos, chemical_content,
                args.seed + repeat * 10000 + budget * 313,
            )
            hyperparameters.append({"repeat": repeat, "known_drugs": budget, **selected})
            for variant, prediction in [
                ("id_oov", id_prediction), ("kernel_content", kernel_prediction)
            ]:
                metrics = evaluate(target, observed, prediction, evaluation)
                rows.append({"repeat": repeat, "known_drugs": budget, "variant": variant, **metrics})
                print(
                    f"repeat={repeat + 1:02d} drugs={budget:02d} variant={variant:14s} "
                    f"rmse={metrics['rmse']:.5f} corr={metrics['mean_row_correlation']:.4f}"
                )
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "kernel_cold_start_results.csv", index=False)
    pd.DataFrame(hyperparameters).to_csv(
        args.output_dir / "kernel_selected_hyperparameters.csv", index=False
    )
    summarize(results, args.output_dir, args.seed)
    audit.update({
        "protocol": "fully unseen evaluation chemicals",
        "model": "RBF kernel ridge on drug-mean PCA response plus learned strain offset",
        "hyperparameter_selection": "leave-one-training-drug-out",
        "evaluation_labels_used_for_selection": False,
    })
    (args.output_dir / "kernel_cold_start_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
