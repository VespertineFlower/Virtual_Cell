from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DRUG_COL = "perturbation_no_concentration"
CONTROL_NAMES = {"Water", "DMSO", "Quality Control"}
FEATURE_VARIANTS = ("morgan", "maccs", "descriptors", "combined")
SHRINKAGES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


def masked_rmse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(truth) & np.isfinite(prediction)
    return float(np.sqrt(np.square(truth[valid] - prediction[valid]).mean()))


def row_mse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(truth) & np.isfinite(prediction)
    squared = np.where(valid, np.square(truth - prediction), 0.0)
    return squared.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


def aggregate_drug_residuals(
    metadata: pd.DataFrame,
    truth: np.ndarray,
    mask: np.ndarray,
    base_prediction: np.ndarray,
    train_rows: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    drugs = sorted(metadata.loc[train_rows, DRUG_COL].astype(str).unique())
    targets = []
    for drug in drugs:
        rows = train_rows & metadata[DRUG_COL].astype(str).eq(drug).to_numpy()
        residual = np.where(mask[rows], truth[rows] - base_prediction[rows], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            targets.append(np.nanmean(residual, axis=0))
    target = np.stack(targets).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fill = np.nanmean(target, axis=0)
    fill = np.nan_to_num(fill, nan=0.0)
    return drugs, np.where(np.isfinite(target), target, fill).astype(np.float32)


def feature_masks(names: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "morgan": np.char.startswith(names, "morgan_"),
        "maccs": np.char.startswith(names, "maccs_"),
        "descriptors": np.char.startswith(names, "descriptor_z_"),
    }


def transform_features(
    raw_train: np.ndarray,
    raw_query: np.ndarray,
    names: np.ndarray,
    variant: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    masks = feature_masks(names)
    blocks_train: list[np.ndarray] = []
    blocks_query: list[np.ndarray] = []
    groups = ("morgan", "maccs", "descriptors") if variant == "combined" else (variant,)
    for group in groups:
        train = raw_train[:, masks[group]].astype(np.float32)
        query = raw_query[:, masks[group]].astype(np.float32)
        if group in {"morgan", "maccs"}:
            components = min(12 if variant == "combined" else 20, len(train) - 1, train.shape[1])
            pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
            train = pca.fit_transform(train)
            query = pca.transform(query)
        scaler = StandardScaler().fit(train)
        blocks_train.append(scaler.transform(train).astype(np.float32))
        blocks_query.append(scaler.transform(query).astype(np.float32))
    return np.concatenate(blocks_train, axis=1), np.concatenate(blocks_query, axis=1)


def rbf_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray,
                alpha: float, gamma: float) -> np.ndarray:
    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = (np.square(left).sum(1)[:, None] + np.square(right).sum(1)[None, :]
                    - 2.0 * left @ right.T)
        return np.exp(-gamma * np.maximum(distance, 0.0))
    gram = kernel(train_x, train_x)
    coefficient = np.linalg.solve(gram + alpha * np.eye(len(train_x)), train_y)
    return kernel(query_x, train_x) @ coefficient


def tanimoto_predict(train_bits: np.ndarray, train_y: np.ndarray,
                     query_bits: np.ndarray, k: int) -> np.ndarray:
    intersection = query_bits @ train_bits.T
    union = query_bits.sum(1)[:, None] + train_bits.sum(1)[None, :] - intersection
    similarity = intersection / np.maximum(union, 1.0)
    neighbors = np.argsort(-similarity, axis=1)[:, :k]
    weights = np.take_along_axis(similarity, neighbors, axis=1)
    weights = np.square(weights) + 1e-6
    selected = train_y[neighbors]
    return (selected * weights[:, :, None]).sum(1) / weights.sum(1)[:, None]


def choose_shrinkage(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    errors = [float(np.square(actual - scale * predicted).mean()) for scale in SHRINKAGES]
    index = int(np.argmin(errors))
    return float(SHRINKAGES[index]), errors[index]


def fit_with_lodo(
    model: str,
    train_x: np.ndarray,
    train_scores: np.ndarray,
    query_x: np.ndarray,
    seed: int,
    response_pca: PCA,
    residual_targets: np.ndarray,
    train_bits: np.ndarray | None = None,
    query_bits: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    n = len(train_x)
    folds = np.arange(n)
    if model == "ridge":
        candidates = [0.1, 1.0, 10.0, 100.0, 1000.0]
        def predict(alpha: float, fit: np.ndarray, query: np.ndarray) -> np.ndarray:
            return Ridge(alpha=alpha).fit(train_x[fit], train_scores[fit]).predict(query)
    elif model == "rbf_kernel":
        candidates = [(a, g) for a in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
                      for g in (0.002, 0.005, 0.01, 0.02, 0.05)]
        def predict(params: tuple[float, float], fit: np.ndarray, query: np.ndarray) -> np.ndarray:
            return rbf_predict(train_x[fit], train_scores[fit], query, *params)
    elif model == "tanimoto_knn":
        if train_bits is None or query_bits is None:
            raise ValueError("Tanimoto kNN requires fingerprint bits")
        candidates = [1, 3, 5, 7]
        def predict(k: int, fit: np.ndarray, query: np.ndarray) -> np.ndarray:
            query_indices = query.astype(np.int64).ravel()
            return tanimoto_predict(train_bits[fit], train_scores[fit], train_bits[query_indices], k)
    else:
        raise ValueError(model)

    best_error = float("inf")
    best_candidate: object = candidates[0]
    best_oof = np.empty_like(train_scores)
    for candidate in candidates:
        oof = np.empty_like(train_scores)
        for held in folds:
            fit = folds != held
            query = np.asarray([[held]]) if model == "tanimoto_knn" else train_x[held:held + 1]
            oof[held] = predict(candidate, fit, query)[0]
        error = float(np.square(oof - train_scores).mean())
        if error < best_error:
            best_error, best_candidate, best_oof = error, candidate, oof.copy()

    oof_residual = response_pca.inverse_transform(best_oof)
    shrinkage, calibrated_error = choose_shrinkage(residual_targets, oof_residual)
    if model == "tanimoto_knn":
        final = tanimoto_predict(train_bits, train_scores, query_bits, int(best_candidate))
    else:
        final = predict(best_candidate, np.ones(n, dtype=bool), query_x)
    return shrinkage * response_pca.inverse_transform(final), {
        "selected_hyperparameter": best_candidate,
        "shrinkage": shrinkage,
        "lodo_score_mse": best_error,
        "lodo_calibrated_mse": calibrated_error,
    }


def fit_random_forest(train_x: np.ndarray, train_scores: np.ndarray,
                      query_x: np.ndarray, seed: int, response_pca: PCA,
                      residual_targets: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    query_predictions = []
    oob_predictions = []
    seeds = [seed + offset * 1009 for offset in range(5)]
    for member_seed in seeds:
        model = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=2,
            max_features="sqrt", random_state=member_seed, n_jobs=-1, oob_score=True,
        )
        model.fit(train_x, train_scores)
        query_predictions.append(model.predict(query_x))
        oob_predictions.append(model.oob_prediction_)
    oob = np.mean(oob_predictions, axis=0).astype(np.float32)
    shrinkage, calibrated_error = choose_shrinkage(
        residual_targets, response_pca.inverse_transform(oob))
    query_prediction = np.mean(query_predictions, axis=0)
    return shrinkage * response_pca.inverse_transform(query_prediction), {
        "selected_hyperparameter": "fixed_low_capacity_5_seed_ensemble",
        "ensemble_seeds": seeds,
        "shrinkage": shrinkage,
        "oob_score_mse": float(np.square(train_scores - oob).mean()),
        "oob_calibrated_mse": calibrated_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Structure-aware correction on official val_chem_only")
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--chemical-features", type=Path,
                        default=Path("external_strain_features/chemical_features.npz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("chemical_structure_official_run"))
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage0 = args.project_dir / "outputs" / "stage0"
    stage1 = args.project_dir / "outputs" / "stage1"
    metadata = pd.read_csv(stage0 / "metadata_aligned.csv").set_index("sample_ID")
    truth = np.load(stage0 / "y_log2.npy", mmap_mode="r")
    mask = np.load(stage0 / "mask.npy", mmap_mode="r")
    sample_ids = np.load(stage0 / "sample_ids.npy", allow_pickle=False).astype(str)
    if metadata.index.astype(str).tolist() != sample_ids.tolist():
        raise ValueError("Stage 0 ordering mismatch")

    sys.path.insert(0, str(args.project_dir / "src"))
    from vc_stage01.features import ConditionFeatureEncoder
    from vc_stage01.mlp import load_checkpoint_predictions

    checkpoint_path = stage1 / "condition_mlp.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = ConditionFeatureEncoder.from_dict(checkpoint["encoder"])
    base_prediction, _ = load_checkpoint_predictions(checkpoint_path, encoder.transform(metadata))

    train_rows = (metadata["split_final"].eq("train")
                  & ~metadata[DRUG_COL].isin(CONTROL_NAMES)).to_numpy()
    validation_rows = (metadata["split_final"].eq("val_chem_only")
                       & ~metadata[DRUG_COL].isin(CONTROL_NAMES)).to_numpy()
    train_drugs, residual_targets = aggregate_drug_residuals(
        metadata, truth, mask, base_prediction, train_rows)
    validation_drugs = sorted(metadata.loc[validation_rows, DRUG_COL].astype(str).unique())

    response_pca = PCA(n_components=min(16, len(train_drugs) - 1),
                       svd_solver="randomized", random_state=args.seed)
    train_scores = response_pca.fit_transform(residual_targets).astype(np.float32)

    archive = np.load(args.chemical_features, allow_pickle=False)
    entity_names = archive["entity_names"].astype(str)
    feature_names = archive["feature_names"].astype(str)
    lookup = {name: i for i, name in enumerate(entity_names)}
    missing = sorted((set(train_drugs) | set(validation_drugs)) - set(lookup))
    if missing:
        raise ValueError(f"Missing chemical structures: {missing}")
    raw = archive["features"].astype(np.float32)
    raw_train = raw[[lookup[name] for name in train_drugs]]
    raw_query = raw[[lookup[name] for name in validation_drugs]]
    morgan = feature_masks(feature_names)["morgan"]
    train_bits, query_bits = raw_train[:, morgan], raw_query[:, morgan]

    predictions: dict[str, np.ndarray] = {"condition_mlp": base_prediction[validation_rows].copy()}
    run_info: dict[str, object] = {}
    query_drug_index = {name: i for i, name in enumerate(validation_drugs)}

    for variant in FEATURE_VARIANTS:
        train_x, query_x = transform_features(
            raw_train, raw_query, feature_names, variant, args.seed)
        models = ("ridge", "random_forest", "rbf_kernel")
        for model_name in models:
            if model_name == "random_forest":
                correction, info = fit_random_forest(
                    train_x, train_scores, query_x, args.seed, response_pca, residual_targets)
            else:
                correction, info = fit_with_lodo(
                    model_name, train_x, train_scores, query_x, args.seed,
                    response_pca, residual_targets)
            correction = correction.astype(np.float32)
            name = f"{model_name}_{variant}"
            sample_correction = np.stack([
                correction[query_drug_index[drug]]
                for drug in metadata.loc[validation_rows, DRUG_COL].astype(str)
            ])
            predictions[name] = base_prediction[validation_rows] + sample_correction
            run_info[name] = {**info, "feature_dimension": int(train_x.shape[1])}

    combined_train, combined_query = transform_features(
        raw_train, raw_query, feature_names, "combined", args.seed)
    correction, info = fit_with_lodo(
        "tanimoto_knn", combined_train, train_scores, combined_query, args.seed,
        response_pca, residual_targets,
        train_bits=train_bits, query_bits=query_bits)
    correction = correction.astype(np.float32)
    sample_correction = np.stack([
        correction[query_drug_index[drug]]
        for drug in metadata.loc[validation_rows, DRUG_COL].astype(str)
    ])
    predictions["tanimoto_knn_morgan"] = base_prediction[validation_rows] + sample_correction
    run_info["tanimoto_knn_morgan"] = {**info, "feature_dimension": int(train_bits.shape[1])}

    validation_truth = np.asarray(truth)[validation_rows]
    validation_mask = np.asarray(mask)[validation_rows]
    validation_meta = metadata.loc[validation_rows]
    metric_rows = []
    per_drug_rows = []
    for name, prediction in predictions.items():
        metric_rows.append({
            "model": name,
            "split": "val_chem_only",
            "samples": int(validation_rows.sum()),
            "valid_values": int(validation_mask.sum()),
            "log2_rmse": masked_rmse(validation_truth, prediction, validation_mask),
        })
        for drug in validation_drugs:
            rows = validation_meta[DRUG_COL].astype(str).eq(drug).to_numpy()
            per_drug_rows.append({
                "model": name, "drug": drug, "samples": int(rows.sum()),
                "log2_rmse": masked_rmse(validation_truth[rows], prediction[rows], validation_mask[rows]),
                "mean_row_mse": float(row_mse(validation_truth[rows], prediction[rows], validation_mask[rows]).mean()),
            })

    metrics = pd.DataFrame(metric_rows).sort_values("log2_rmse")
    per_drug = pd.DataFrame(per_drug_rows)
    base_by_drug = per_drug.loc[per_drug["model"].eq("condition_mlp")].set_index("drug")
    rng = np.random.default_rng(args.seed)
    paired_rows = []
    for name in predictions:
        if name == "condition_mlp":
            continue
        candidate = per_drug.loc[per_drug["model"].eq(name)].set_index("drug")
        gains = (base_by_drug.loc[validation_drugs, "log2_rmse"]
                 - candidate.loc[validation_drugs, "log2_rmse"]).to_numpy()
        boot = np.asarray([
            rng.choice(gains, size=len(gains), replace=True).mean() for _ in range(20000)
        ])
        paired_rows.append({
            "model": name,
            "mean_drug_rmse_improvement": float(gains.mean()),
            "median_drug_rmse_improvement": float(np.median(gains)),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "drug_wins": int((gains > 0).sum()),
            "drugs": len(gains),
        })
    paired = pd.DataFrame(paired_rows).sort_values(
        "mean_drug_rmse_improvement", ascending=False)
    metrics.to_csv(args.output_dir / "official_val_chem_metrics.csv", index=False)
    per_drug.to_csv(args.output_dir / "official_val_chem_per_drug.csv", index=False)
    paired.to_csv(args.output_dir / "official_val_chem_paired.csv", index=False)
    (args.output_dir / "official_val_chem_audit.json").write_text(json.dumps({
        "protocol": "official split_final=val_chem_only; absolute log2 abundance",
        "base_model": str(checkpoint_path.resolve()),
        "correction_target": "train-only per-chemical mean Condition-MLP residual",
        "selection": "training-chemical leave-one-drug-out or random-forest OOB only",
        "validation_labels_used_for_selection": False,
        "train_chemicals": train_drugs,
        "validation_chemicals": validation_drugs,
        "response_pca_components": int(response_pca.n_components_),
        "feature_variants": list(FEATURE_VARIANTS),
        "models": run_info,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
