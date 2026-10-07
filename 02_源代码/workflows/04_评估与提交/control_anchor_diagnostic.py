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

from chemical_structure_official import fit_with_lodo, transform_features


DRUG_COL = "perturbation_no_concentration"


def masked_rmse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(truth) & np.isfinite(prediction)
    return float(np.sqrt(np.square(truth[valid] - prediction[valid]).mean()))


def aggregate_delta(metadata: pd.DataFrame, rows: np.ndarray, truth: np.ndarray,
                    controls: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    delta = np.where(np.isfinite(truth) & np.isfinite(controls), truth - controls, np.nan)
    drugs = sorted(metadata.iloc[rows][DRUG_COL].astype(str).unique())
    means = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_delta = np.nanmean(delta, axis=0)
    global_delta = np.nan_to_num(global_delta, nan=0.0).astype(np.float32)
    row_drugs = metadata.iloc[rows][DRUG_COL].astype(str).to_numpy()
    for drug in drugs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            value = np.nanmean(delta[row_drugs == drug], axis=0)
        means.append(np.where(np.isfinite(value), value, global_delta))
    return drugs, np.stack(means).astype(np.float32), global_delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle and deployable control-anchor diagnostics")
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--chemical-features", type=Path,
                        default=Path("external_strain_features/chemical_features.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("control_anchor_diagnostic_run"))
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.project_dir / "src"))
    from vc_stage01.baselines import matched_control_predictions, treatment_rows
    from vc_stage01.constants import CONTROL_NAMES, EXCLUDED_TREATMENTS, VALIDATION_SPLITS
    from vc_stage01.features import ConditionFeatureEncoder
    from vc_stage01.mlp import load_checkpoint_predictions
    from vc_stage01.official_proxy import fit_proxy_reference, score_official_proxy
    from vc_stage01.stage2 import strict_matched_controls

    stage0 = args.project_dir / "outputs" / "stage0"
    stage1 = args.project_dir / "outputs" / "stage1"
    metadata = pd.read_csv(stage0 / "metadata_aligned.csv").set_index("sample_ID")
    truth = np.load(stage0 / "y_log2.npy", mmap_mode="r")
    mask = np.load(stage0 / "mask.npy", mmap_mode="r")
    ids = np.load(stage0 / "sample_ids.npy", allow_pickle=False).astype(str)
    if metadata.index.astype(str).tolist() != ids.tolist():
        raise ValueError("Stage 0 sample ordering mismatch")
    train = metadata["split_final"].eq("train").to_numpy()
    validation = metadata["split_final"].isin(VALIDATION_SPLITS).to_numpy()

    checkpoint_path = stage1 / "condition_mlp.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = ConditionFeatureEncoder.from_dict(checkpoint["encoder"])
    condition_prediction, _ = load_checkpoint_predictions(
        checkpoint_path, encoder.transform(metadata))

    train_treatments = train & ~metadata[DRUG_COL].isin(EXCLUDED_TREATMENTS).to_numpy()
    train_pairs, train_controls, _ = strict_matched_controls(
        metadata, truth, train_treatments, train)
    train_drugs, drug_delta, global_delta = aggregate_delta(
        metadata, train_pairs, np.asarray(truth)[train_pairs], train_controls)

    response_pca = PCA(n_components=min(24, len(train_drugs) - 1),
                       svd_solver="randomized", random_state=args.seed)
    train_scores = response_pca.fit_transform(drug_delta).astype(np.float32)
    archive = np.load(args.chemical_features, allow_pickle=False)
    entity_names = archive["entity_names"].astype(str)
    feature_names = archive["feature_names"].astype(str)
    feature_lookup = {name: i for i, name in enumerate(entity_names)}
    validation_drugs = sorted(metadata.loc[validation, DRUG_COL].astype(str).unique())
    query_drugs = [drug for drug in validation_drugs if drug not in train_drugs
                   and drug not in CONTROL_NAMES and drug != "Quality Control"]
    missing = sorted((set(train_drugs) | set(query_drugs)) - set(feature_lookup))
    if missing:
        raise ValueError(f"Missing chemical features: {missing}")
    raw = archive["features"].astype(np.float32)
    raw_train = raw[[feature_lookup[name] for name in train_drugs]]
    raw_query = raw[[feature_lookup[name] for name in query_drugs]]
    train_x, query_x = transform_features(
        raw_train, raw_query, feature_names, "combined", args.seed)
    if query_drugs:
        structure_delta, structure_info = fit_with_lodo(
            "rbf_kernel", train_x, train_scores, query_x, args.seed,
            response_pca, drug_delta)
    else:
        structure_delta = np.empty((0, truth.shape[1]), dtype=np.float32)
        structure_info = {}
    delta_lookup = dict(zip(train_drugs, drug_delta))
    delta_lookup.update(dict(zip(query_drugs, structure_delta.astype(np.float32))))

    # Generate a deployable pseudo-control by changing only the chemical field
    # to each training control identity and averaging the two MLP predictions.
    synthetic_predictions = []
    for control_name in sorted(CONTROL_NAMES):
        synthetic = metadata.copy()
        synthetic[DRUG_COL] = control_name
        values, _ = load_checkpoint_predictions(checkpoint_path, encoder.transform(synthetic))
        synthetic_predictions.append(values)
    pseudo_control = np.mean(synthetic_predictions, axis=0).astype(np.float32)

    validation_positions = np.flatnonzero(validation)
    validation_metadata = metadata.loc[validation].copy()
    train_strain_names = sorted(metadata.loc[train, "Strains"].astype(str).unique())
    transferred_controls: dict[str, np.ndarray] = {}
    for strain_name in train_strain_names:
        control_predictions = []
        for control_name in sorted(CONTROL_NAMES):
            synthetic = validation_metadata.copy()
            unknown = ~synthetic["Strains"].astype(str).isin(train_strain_names)
            synthetic.loc[unknown, "Strains"] = strain_name
            synthetic[DRUG_COL] = control_name
            values, _ = load_checkpoint_predictions(
                checkpoint_path, encoder.transform(synthetic))
            control_predictions.append(values)
        transferred_controls[strain_name] = np.mean(control_predictions, axis=0).astype(np.float32)

    def strain_weights(path: Path) -> tuple[str, dict[str, np.ndarray]]:
        bundle = np.load(path, allow_pickle=False)
        names = bundle["entity_names"].astype(str)
        features = bundle["features"].astype(np.float64)
        entity = dict(zip(names, features))
        train_features = np.stack([entity[name] for name in train_strain_names])
        pairwise = np.linalg.norm(train_features[:, None] - train_features[None, :], axis=2)
        nonzero = pairwise[pairwise > 0]
        scale = float(np.median(nonzero)) if len(nonzero) else 1.0
        result = {}
        for query in validation_metadata["Strains"].astype(str).unique():
            if query in train_strain_names:
                weight = np.zeros(len(train_strain_names), dtype=np.float32)
                weight[train_strain_names.index(query)] = 1.0
            else:
                distance = np.linalg.norm(train_features - entity[query], axis=1)
                weight = np.exp(-np.square(distance) / (2.0 * max(scale, 1e-6) ** 2))
                weight = (weight / weight.sum()).astype(np.float32)
            result[query] = weight
        unknown_names = [name for name in result if name not in train_strain_names]
        nearest = train_strain_names[int(np.argmax(result[unknown_names[0]]))] if unknown_names else "none"
        return nearest, result

    sequence_nearest, sequence_weights = strain_weights(
        args.project_dir / "outputs" / "features" / "strain_sequence_embeddings.npz")
    population_nearest, population_weights = strain_weights(
        args.project_dir / "outputs" / "features" / "strain_features.npz")

    def mix_controls(weights: dict[str, np.ndarray]) -> np.ndarray:
        stack = np.stack([transferred_controls[name] for name in train_strain_names])
        return np.stack([
            np.sum(stack[:, row] * weights[strain][:, None], axis=0)
            for row, strain in enumerate(validation_metadata["Strains"].astype(str))
        ]).astype(np.float32)

    strain_control_candidates = {
        "oov": pseudo_control[validation],
        "uniform": np.mean(np.stack(list(transferred_controls.values())), axis=0),
        "nearest_sequence": transferred_controls[sequence_nearest],
        "kernel_sequence": mix_controls(sequence_weights),
        "nearest_population": transferred_controls[population_nearest],
        "kernel_population": mix_controls(population_weights),
    }

    routed_delta = np.stack([
        delta_lookup.get(drug, global_delta)
        for drug in metadata[DRUG_COL].astype(str)
    ]).astype(np.float32)
    pseudo_routed = pseudo_control + routed_delta
    pseudo_global = pseudo_control + global_delta.reshape(1, -1)

    # Drug means are reliable for seen drugs; unseen drugs use the global
    # response because the structure route failed its train-only diagnostic.
    conservative_delta = np.stack([
        dict(zip(train_drugs, drug_delta)).get(drug, global_delta)
        for drug in metadata[DRUG_COL].astype(str)
    ]).astype(np.float32)

    strain_transfer_rows: list[dict[str, object]] = []
    for split in VALIDATION_SPLITS:
        candidates = treatment_rows(metadata, split)
        rows, controls, _ = matched_control_predictions(metadata, truth, candidates)
        local = np.searchsorted(validation_positions, rows)
        split_truth = np.asarray(truth)[rows]
        split_mask = np.asarray(mask)[rows] & np.isfinite(controls)
        for name, control_prediction in strain_control_candidates.items():
            anchor = control_prediction[local]
            prediction = anchor + conservative_delta[rows]
            strain_transfer_rows.append({
                "control_transfer": name, "split": split, "samples": len(rows),
                "anchor_rmse": masked_rmse(
                    controls, anchor, np.isfinite(controls) & np.isfinite(anchor)),
                "prediction_rmse": masked_rmse(split_truth, prediction, split_mask),
            })
    strain_transfer = pd.DataFrame(strain_transfer_rows)
    strain_transfer.to_csv(args.output_dir / "strain_transfer_metrics.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    for split in VALIDATION_SPLITS:
        candidates = treatment_rows(metadata, split)
        rows, controls, _ = matched_control_predictions(metadata, truth, candidates)
        split_truth = np.asarray(truth)[rows]
        split_mask = np.asarray(mask)[rows] & np.isfinite(controls)
        oracle_global = controls + global_delta.reshape(1, -1)
        oracle_routed = controls + routed_delta[rows]
        models = {
            "matched_control": controls,
            "oracle_global_delta": oracle_global,
            "oracle_routed_delta": oracle_routed,
            "pseudo_control_only": pseudo_control[rows],
            "pseudo_global_delta": pseudo_global[rows],
            "pseudo_routed_delta": pseudo_routed[rows],
            "condition_mlp": condition_prediction[rows],
        }
        for name, prediction in models.items():
            metric_rows.append({
                "model": name, "split": split, "samples": len(rows),
                "valid_values": int(split_mask.sum()),
                "log2_rmse": masked_rmse(split_truth, prediction, split_mask),
            })
        control_mask = np.isfinite(controls) & np.isfinite(pseudo_control[rows])
        anchor_rows.append({
            "split": split, "samples": len(rows),
            "pseudo_control_vs_true_control_rmse": masked_rmse(
                controls, pseudo_control[rows], control_mask),
        })

    metrics = pd.DataFrame(metric_rows)
    anchors = pd.DataFrame(anchor_rows)
    metrics.to_csv(args.output_dir / "matched_subset_metrics.csv", index=False)
    anchors.to_csv(args.output_dir / "control_anchor_metrics.csv", index=False)

    reference = fit_proxy_reference(metadata, truth, train)
    proxy_records = []
    proxy_summaries = {}
    for name, prediction in {
        "condition_mlp": condition_prediction,
        "pseudo_control_only": pseudo_control,
        "pseudo_routed_delta": pseudo_routed,
    }.items():
        records, summary = score_official_proxy(
            metadata, truth, prediction, train, reference)
        proxy_records.extend({"model": name, **record} for record in records)
        proxy_summaries[name] = summary
    pd.DataFrame(proxy_records).to_csv(args.output_dir / "official_proxy_by_split.csv", index=False)
    (args.output_dir / "audit.json").write_text(json.dumps({
        "protocol": "train-only strict delta; oracle and deployable pseudo-control diagnostics",
        "train_pairs": int(len(train_pairs)),
        "train_drugs": train_drugs,
        "query_drugs": query_drugs,
        "structure_model": structure_info,
        "strain_transfer": {
            "train_strains": train_strain_names,
            "sequence_nearest_for_unknown": sequence_nearest,
            "population_nearest_for_unknown": population_nearest,
            "selection_uses_validation_labels": False,
        },
        "condition_checkpoint": str(checkpoint_path.resolve()),
        "validation_labels_used_for_training_or_selection": False,
        "oracle_models_are_submission_eligible": False,
        "pseudo_models_are_submission_eligible": True,
        "official_proxy_reference": reference.summary(),
        "official_proxy_summaries": proxy_summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMatched-subset RMSE:\n", metrics.pivot(
        index="model", columns="split", values="log2_rmse").to_string())
    print("\nPseudo-control anchor error:\n", anchors.to_string(index=False))
    print("\nStrain-transfer prediction RMSE:\n", strain_transfer.pivot(
        index="control_transfer", columns="split", values="prediction_rmse").to_string())
    print("\nStrain-transfer anchor RMSE:\n", strain_transfer.pivot(
        index="control_transfer", columns="split", values="anchor_rmse").to_string())
    print("\nOfficial proxy summaries:\n", json.dumps(
        {name: value.get("proxy_score") for name, value in proxy_summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
