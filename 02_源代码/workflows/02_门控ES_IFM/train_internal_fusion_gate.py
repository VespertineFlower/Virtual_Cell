from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline10_es_ifm_v4 import V2Config, fit_full, predict_rows, prepare_run
from baseline8_es_ifm_v2 import EXCLUDED_TREATMENTS, pseudo_s3_masks
from es_ifm_mlp_fusion import load_selected_csv, masked_rmse, oracle_alpha


VALIDATION_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit MLP/ES-IFM fusion gates on train-only pseudo OOV data")
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("train_internal_fusion_gate_run"))
    parser.add_argument("--mlp-dir", type=Path, default=Path("outputs/stage1"))
    parser.add_argument(
        "--es-ifm", type=Path, default=Path("prediction_es_ifm_v4_no_episodic.csv")
    )
    parser.add_argument(
        "--es-ifm-flow",
        type=Path,
        default=Path("prediction_es_ifm_v4_no_episodic_no_residual.csv"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.project_dir / "src"))
    from vc_stage01.features import ConditionFeatureEncoder
    from vc_stage01.mlp import MLPConfig, train_condition_mlp
    from vc_stage01.official_proxy import fit_proxy_reference, score_official_proxy

    stage0 = args.project_dir / "outputs" / "stage0"
    metadata = pd.read_csv(stage0 / "metadata_aligned.csv").set_index("sample_ID")
    metadata.index = metadata.index.astype(str)
    sample_ids = np.load(stage0 / "sample_ids.npy", allow_pickle=False).astype(str)
    truth = np.load(stage0 / "y_log2.npy", mmap_mode="r")
    observed = np.load(stage0 / "mask.npy", mmap_mode="r").astype(bool)
    protein_names = pd.read_csv(stage0 / "feature_contract.csv")["protein"].astype(str).tolist()
    y_frame = pd.DataFrame(np.asarray(truth), index=metadata.index, columns=protein_names)

    pseudo_fit, pseudo_both, pseudo_info = pseudo_s3_masks(metadata)
    official_train = metadata["split_final"].eq("train").to_numpy()
    treated = ~metadata["perturbation_no_concentration"].isin(EXCLUDED_TREATMENTS).to_numpy()
    held_strain = str(pseudo_info["held_strain"])
    held_drugs = set(str(value) for value in pseudo_info["held_drugs"])
    strain_held = metadata["Strains"].astype(str).eq(held_strain).to_numpy()
    drug_held = metadata["perturbation_no_concentration"].astype(str).isin(held_drugs).to_numpy()
    pseudo_splits = {
        "pseudo_chem_only": official_train & treated & ~strain_held & drug_held,
        "pseudo_strain_only": official_train & treated & strain_held & ~drug_held,
        "pseudo_both": pseudo_both,
    }
    pseudo_eval = np.logical_or.reduce(list(pseudo_splits.values()))

    # The early-stopping rows are disjoint from both pseudo OOV evaluation and MLP fitting.
    fit_positions = np.flatnonzero(pseudo_fit)
    inner_validation = np.zeros(len(metadata), dtype=bool)
    inner_validation[fit_positions[::10]] = True
    mlp_fit = pseudo_fit & ~inner_validation
    encoder = ConditionFeatureEncoder.fit(metadata.iloc[np.flatnonzero(mlp_fit)])
    features = encoder.transform(metadata)
    protein_mean = np.nanmean(np.asarray(truth)[mlp_fit], axis=0).astype(np.float32)
    mlp_prediction, mlp_history, mlp_info = train_condition_mlp(
        features=features,
        y_log2=truth,
        observation_mask=observed,
        train_rows=mlp_fit,
        validation_rows=inner_validation,
        protein_mean_vector=protein_mean,
        config=MLPConfig(seed=20260815),
        output_dir=args.output_dir / "pseudo_mlp",
        encoder_info=encoder.to_dict(),
        protein_names=protein_names,
    )

    es_config = V2Config(
        latent_dim=128,
        hidden_dim=192,
        batch_size=192,
        quadrature_steps=3,
        inference_steps=8,
        seed=20260814,
    )
    es_run = prepare_run(metadata, y_frame, observed, pseudo_fit, pseudo_eval, es_config)
    es_model = fit_full(es_run, es_config, epochs=4, episodic=False)
    pseudo_positions = np.flatnonzero(pseudo_eval)
    es_full = predict_rows(es_model, es_run, pseudo_positions, es_config, use_residual=True)
    es_flow = predict_rows(es_model, es_run, pseudo_positions, es_config, use_residual=False)
    pseudo_mlp = mlp_prediction[pseudo_positions]
    pseudo_truth = np.asarray(truth)[pseudo_positions]
    pseudo_observed = observed[pseudo_positions]

    local_lookup = {position: local for local, position in enumerate(pseudo_positions)}
    gates: dict[str, dict[str, float]] = {"full": {}, "flow": {}}
    gate_records = []
    for branch, branch_prediction in (("full", es_full), ("flow", es_flow)):
        direction = branch_prediction - pseudo_mlp
        global_alpha = oracle_alpha(pseudo_truth, pseudo_mlp, direction, pseudo_observed)
        gates[branch]["global"] = global_alpha
        gate_records.append({"branch": branch, "scope": "global", "alpha": global_alpha})
        for split, split_mask in pseudo_splits.items():
            local = np.asarray([local_lookup[pos] for pos in np.flatnonzero(split_mask)])
            alpha = oracle_alpha(
                pseudo_truth[local], pseudo_mlp[local], direction[local], pseudo_observed[local]
            )
            gates[branch][split] = alpha
            gate_records.append({"branch": branch, "scope": split, "alpha": alpha})

    gate_frame = pd.DataFrame(gate_records)
    gate_frame.to_csv(args.output_dir / "train_only_gates.csv", index=False)

    validation = metadata["split_final"].isin(VALIDATION_SPLITS).to_numpy()
    validation_positions = np.flatnonzero(validation)
    validation_ids = sample_ids[validation_positions]
    saved_ids = np.load(
        args.mlp_dir / "condition_mlp_validation_sample_ids.npy", allow_pickle=False
    ).astype(str)
    saved_mlp = np.load(
        args.mlp_dir / "condition_mlp_validation_predictions.npy", mmap_mode="r"
    )
    saved_lookup = {sample_id: row for row, sample_id in enumerate(saved_ids)}
    validation_mlp = np.asarray(
        saved_mlp[[saved_lookup[sample_id] for sample_id in validation_ids]], dtype=np.float32
    )
    validation_full = load_selected_csv(
        args.es_ifm, sample_ids, validation_positions, protein_names
    )
    validation_flow = load_selected_csv(
        args.es_ifm_flow, sample_ids, validation_positions, protein_names
    )
    validation_split = metadata.iloc[validation_positions]["split_final"].astype(str).to_numpy()

    candidates = {"condition_mlp": validation_mlp}
    route_map = {
        "val_chem_only": "pseudo_chem_only",
        "val_strain_only": "pseudo_strain_only",
        "val_both": "pseudo_both",
    }
    for branch, branch_prediction in (("full", validation_full), ("flow", validation_flow)):
        direction = branch_prediction - validation_mlp
        global_alpha = gates[branch]["global"]
        candidates[f"train_gate_{branch}_global"] = validation_mlp + global_alpha * direction
        routed = validation_mlp.copy()
        for split in VALIDATION_SPLITS:
            rows = validation_split == split
            alpha = gates[branch].get(route_map.get(split, ""), global_alpha)
            routed[rows] = validation_mlp[rows] + alpha * direction[rows]
        candidates[f"train_gate_{branch}_routed"] = routed
    candidates["validation_selected_full_0.20"] = (
        0.8 * validation_mlp + 0.2 * validation_full
    )

    rmse_records = []
    validation_truth = np.asarray(truth)[validation_positions]
    validation_observed = observed[validation_positions]
    validation_treated = ~metadata.iloc[validation_positions][
        "perturbation_no_concentration"
    ].isin(EXCLUDED_TREATMENTS).to_numpy()
    for name, prediction in candidates.items():
        for split in VALIDATION_SPLITS:
            rows = validation_treated & (validation_split == split)
            rmse_records.append(
                {
                    "model": name,
                    "split": split,
                    "samples": int(rows.sum()),
                    "rmse": masked_rmse(
                        validation_truth[rows], prediction[rows], validation_observed[rows]
                    ),
                }
            )
    pd.DataFrame(rmse_records).to_csv(args.output_dir / "validation_rmse.csv", index=False)

    reference = fit_proxy_reference(metadata, truth, official_train)
    proxy_records = []
    proxy_split_records = []
    full_prediction = np.zeros_like(truth, dtype=np.float32)
    for name, prediction in candidates.items():
        full_prediction[validation_positions] = prediction
        records, summary = score_official_proxy(
            metadata, truth, full_prediction, official_train, reference
        )
        proxy_records.append({"model": name, **summary})
        proxy_split_records.extend({"model": name, **record} for record in records)
    proxy_frame = pd.DataFrame(proxy_records).sort_values("proxy_score", ascending=False)
    proxy_frame.to_csv(args.output_dir / "validation_proxy.csv", index=False)
    pd.DataFrame(proxy_split_records).to_csv(
        args.output_dir / "validation_proxy_by_split.csv", index=False
    )

    audit = {
        "pseudo_split": pseudo_info,
        "pseudo_counts": {name: int(mask.sum()) for name, mask in pseudo_splits.items()},
        "mlp_fit_samples": int(mlp_fit.sum()),
        "mlp_inner_validation_samples": int(inner_validation.sum()),
        "mlp_training": mlp_info,
        "es_ifm_epochs": 4,
        "gates": gates,
        "best_validation_model": proxy_frame.iloc[0].to_dict(),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Train-only gates:")
    print(gate_frame.to_string(index=False))
    print("\nOfficial validation proxy:")
    print(proxy_frame[["model", "proxy_score", "module1_fc", "module2_fidelity"]].to_string(index=False))
    print(f"\nsaved: {args.output_dir}")


if __name__ == "__main__":
    main()
