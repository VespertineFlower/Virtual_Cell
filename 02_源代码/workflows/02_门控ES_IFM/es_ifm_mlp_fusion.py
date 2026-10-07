from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


VALIDATION_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")
EXCLUDED_TREATMENTS = {"Water", "DMSO", "Quality Control"}
GRID = (-0.25, -0.10, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0)
HEAD_GRID = (-1.0, -0.50, -0.25, 0.0, 0.25, 0.50, 1.0)


def load_selected_csv(
    path: Path,
    metadata_ids: np.ndarray,
    selected_positions: np.ndarray,
    protein_names: list[str],
) -> np.ndarray:
    selected = set(int(value) for value in selected_positions)
    frame = pd.read_csv(
        path,
        index_col="sample_ID",
        usecols=["sample_ID", *protein_names],
        skiprows=lambda line: line > 0 and (line - 1) not in selected,
    )
    expected = metadata_ids[selected_positions]
    frame.index = frame.index.astype(str)
    if set(frame.index) != set(expected):
        missing = sorted(set(expected) - set(frame.index))[:5]
        raise ValueError(f"{path} is not aligned with metadata; missing IDs: {missing}")
    return frame.loc[expected].to_numpy(np.float32)


def masked_rmse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(truth) & np.isfinite(prediction)
    return float(np.sqrt(np.square(truth[valid] - prediction[valid]).mean()))


def oracle_alpha(
    truth: np.ndarray,
    base: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
) -> float:
    valid = mask & np.isfinite(truth) & np.isfinite(base) & np.isfinite(direction)
    d = direction[valid].astype(np.float64)
    residual = (truth - base)[valid].astype(np.float64)
    denominator = float(np.dot(d, d))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(d, residual) / denominator, -2.0, 2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Condition MLP + ES-IFM-v4 fusion diagnostic")
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("es_ifm_mlp_fusion_run"))
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
    from vc_stage01.official_proxy import fit_proxy_reference, score_official_proxy

    stage0 = args.project_dir / "outputs" / "stage0"
    metadata = pd.read_csv(stage0 / "metadata_aligned.csv").set_index("sample_ID")
    metadata.index = metadata.index.astype(str)
    truth_all = np.load(stage0 / "y_log2.npy", mmap_mode="r")
    mask_all = np.load(stage0 / "mask.npy", mmap_mode="r").astype(bool)
    sample_ids = np.load(stage0 / "sample_ids.npy", allow_pickle=False).astype(str)
    protein_names = pd.read_csv(stage0 / "feature_contract.csv")["protein"].astype(str).tolist()
    if metadata.index.tolist() != sample_ids.tolist():
        raise ValueError("Stage 0 sample ordering mismatch")

    validation_mask = metadata["split_final"].isin(VALIDATION_SPLITS).to_numpy()
    validation_positions = np.flatnonzero(validation_mask)
    validation_ids = sample_ids[validation_positions]
    saved_mlp_ids = np.load(
        args.mlp_dir / "condition_mlp_validation_sample_ids.npy", allow_pickle=False
    ).astype(str)
    saved_mlp = np.load(
        args.mlp_dir / "condition_mlp_validation_predictions.npy", mmap_mode="r"
    )
    mlp_lookup = {sample_id: row for row, sample_id in enumerate(saved_mlp_ids)}
    mlp_rows = np.asarray([mlp_lookup[sample_id] for sample_id in validation_ids])
    mlp = np.asarray(saved_mlp[mlp_rows], dtype=np.float32)
    es_ifm = load_selected_csv(
        args.es_ifm, sample_ids, validation_positions, protein_names
    )
    es_ifm_flow = load_selected_csv(
        args.es_ifm_flow, sample_ids, validation_positions, protein_names
    )
    truth = np.asarray(truth_all[validation_positions], dtype=np.float32)
    observed = np.asarray(mask_all[validation_positions], dtype=bool)

    split_values = metadata.iloc[validation_positions]["split_final"].astype(str).to_numpy()
    treated = ~metadata.iloc[validation_positions]["perturbation_no_concentration"].isin(
        EXCLUDED_TREATMENTS
    ).to_numpy()
    full_direction = es_ifm - mlp
    flow_direction = es_ifm_flow - mlp
    residual_head = es_ifm - es_ifm_flow

    candidates: dict[str, np.ndarray] = {
        "condition_mlp": mlp,
        "es_ifm_v4": es_ifm,
        "es_ifm_v4_flow": es_ifm_flow,
    }
    for alpha in GRID:
        candidates[f"convex_alpha_{alpha:+.2f}"] = mlp + alpha * full_direction
        candidates[f"structured_flow_alpha_{alpha:+.2f}"] = mlp + alpha * flow_direction
    for alpha in HEAD_GRID:
        candidates[f"residual_head_alpha_{alpha:+.2f}"] = mlp + alpha * residual_head

    oracle_records = []
    for direction_name, direction in (
        ("full_es_ifm_direction", full_direction),
        ("structured_flow_direction", flow_direction),
        ("es_ifm_residual_head", residual_head),
    ):
        alpha = oracle_alpha(truth, mlp, direction, observed & treated[:, None])
        name = f"oracle_global_{direction_name}"
        candidates[name] = mlp + alpha * direction
        oracle_records.append({"scope": "all", "direction": direction_name, "alpha": alpha})
        routed = mlp.copy()
        for split in VALIDATION_SPLITS:
            rows = treated & (split_values == split)
            split_alpha = oracle_alpha(truth[rows], mlp[rows], direction[rows], observed[rows])
            routed[rows] = mlp[rows] + split_alpha * direction[rows]
            oracle_records.append(
                {"scope": split, "direction": direction_name, "alpha": split_alpha}
            )
        candidates[f"oracle_routed_{direction_name}"] = routed

    rmse_records = []
    for name, prediction in candidates.items():
        for split in VALIDATION_SPLITS:
            rows = treated & (split_values == split)
            rmse_records.append(
                {
                    "model": name,
                    "split": split,
                    "samples": int(rows.sum()),
                    "rmse": masked_rmse(truth[rows], prediction[rows], observed[rows]),
                }
            )
    rmse_frame = pd.DataFrame(rmse_records)
    rmse_frame.to_csv(args.output_dir / "rmse_by_split.csv", index=False)

    train_rows = metadata["split_final"].eq("train").to_numpy()
    reference = fit_proxy_reference(metadata, truth_all, train_rows)
    proxy_records = []
    proxy_split_records = []
    full_predictions = np.zeros_like(truth_all, dtype=np.float32)
    for name, prediction in candidates.items():
        full_predictions[validation_positions] = prediction
        records, summary = score_official_proxy(
            metadata, truth_all, full_predictions, train_rows, reference
        )
        proxy_records.append({"model": name, **summary})
        proxy_split_records.extend({"model": name, **record} for record in records)
    proxy_frame = pd.DataFrame(proxy_records).sort_values("proxy_score", ascending=False)
    proxy_frame.to_csv(args.output_dir / "official_proxy_comparison.csv", index=False)
    pd.DataFrame(proxy_split_records).to_csv(
        args.output_dir / "official_proxy_by_split.csv", index=False
    )
    pd.DataFrame(oracle_records).to_csv(args.output_dir / "oracle_weights.csv", index=False)

    best_deployable = proxy_frame.loc[~proxy_frame["model"].str.startswith("oracle")].iloc[0]
    audit = {
        "best_deployable": best_deployable.to_dict(),
        "grid": list(GRID),
        "residual_head_grid": list(HEAD_GRID),
        "warning": "oracle rows use validation truth and are diagnostic upper bounds only",
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(proxy_frame[["model", "proxy_score", "module1_fc", "module2_fidelity"]].to_string(index=False))
    print("\nOracle scalar weights:")
    print(pd.DataFrame(oracle_records).to_string(index=False))
    print(f"\nsaved: {args.output_dir}")


if __name__ == "__main__":
    main()
