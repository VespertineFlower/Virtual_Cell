from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vc_stage01.baselines import (  # noqa: E402
    matched_control_predictions,
    protein_mean,
    treatment_rows,
)
from vc_stage01.constants import EXCLUDED_TREATMENTS, VALIDATION_SPLITS  # noqa: E402
from vc_stage01.features import ConditionFeatureEncoder  # noqa: E402
from vc_stage01.metrics import masked_regression_metrics  # noqa: E402
from vc_stage01.mlp import MLPConfig, train_condition_mlp  # noqa: E402


def evaluate(
    model_name: str,
    scope: str,
    split: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {
        "model": model_name,
        "evaluation_scope": scope,
        "split": split,
    }
    result.update(masked_regression_metrics(truth, prediction, mask))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 GOAI baselines.")
    parser.add_argument("--stage0-dir", type=Path, default=PROJECT_ROOT / "outputs" / "stage0")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "stage1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    metadata = pd.read_csv(args.stage0_dir / "metadata_aligned.csv").set_index("sample_ID")
    contract = pd.read_csv(args.stage0_dir / "feature_contract.csv")
    y_log2 = np.load(args.stage0_dir / "y_log2.npy", mmap_mode="r")
    mask = np.load(args.stage0_dir / "mask.npy", mmap_mode="r")
    sample_ids = np.load(args.stage0_dir / "sample_ids.npy", allow_pickle=False).astype(str)
    if metadata.index.astype(str).tolist() != sample_ids.tolist():
        raise ValueError("Stage 0 sample ordering contract failed")
    if y_log2.shape != mask.shape or y_log2.shape[1] != len(contract):
        raise ValueError("Stage 0 array/feature contract mismatch")

    train_rows = metadata["split_final"].eq("train").to_numpy()
    validation_rows = metadata["split_final"].isin(VALIDATION_SPLITS).to_numpy()
    mean_vector = protein_mean(y_log2, train_rows)

    encoder = ConditionFeatureEncoder.fit(metadata.loc[train_rows])
    features = encoder.transform(metadata)
    config = MLPConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlp_prediction, _, mlp_run_info = train_condition_mlp(
        features=features,
        y_log2=y_log2,
        observation_mask=mask,
        train_rows=train_rows,
        validation_rows=validation_rows,
        protein_mean_vector=mean_vector,
        config=config,
        output_dir=args.output_dir,
        encoder_info=encoder.to_dict(),
        protein_names=contract["protein"].astype(str).tolist(),
    )

    all_treatment_records: list[dict[str, object]] = []
    matched_records: list[dict[str, object]] = []
    coverage_records: list[dict[str, object]] = []
    mean_prediction_all = np.broadcast_to(mean_vector, y_log2.shape)

    for split in VALIDATION_SPLITS:
        candidate = treatment_rows(metadata, split)
        positions = np.flatnonzero(candidate)
        all_treatment_records.append(
            evaluate(
                "protein_mean",
                "all_treatments",
                split,
                np.asarray(y_log2)[positions],
                mean_prediction_all[positions],
                np.asarray(mask)[positions],
            )
        )
        all_treatment_records.append(
            evaluate(
                "condition_mlp",
                "all_treatments",
                split,
                np.asarray(y_log2)[positions],
                mlp_prediction[positions],
                np.asarray(mask)[positions],
            )
        )

        matched_rows, control_prediction, control_counts = matched_control_predictions(
            metadata, y_log2, candidate
        )
        coverage_records.append(
            {
                "split": split,
                "candidate_treatments": int(candidate.sum()),
                "matched_samples": int(len(matched_rows)),
                "coverage": float(len(matched_rows) / max(int(candidate.sum()), 1)),
                "mean_controls_per_sample": float(control_counts.mean()),
            }
        )
        matched_truth = np.asarray(y_log2)[matched_rows]
        matched_mask = np.asarray(mask)[matched_rows] & np.isfinite(control_prediction)
        for name, prediction in (
            ("protein_mean", mean_prediction_all[matched_rows]),
            ("matched_control", control_prediction),
            ("condition_mlp", mlp_prediction[matched_rows]),
        ):
            matched_records.append(
                evaluate(
                    name,
                    "matched_control_subset",
                    split,
                    matched_truth,
                    prediction,
                    matched_mask,
                )
            )

    all_frame = pd.DataFrame(all_treatment_records)
    matched_frame = pd.DataFrame(matched_records)
    coverage_frame = pd.DataFrame(coverage_records)
    all_frame.to_csv(args.output_dir / "metrics_all_treatments.csv", index=False)
    matched_frame.to_csv(args.output_dir / "metrics_matched_subset.csv", index=False)
    coverage_frame.to_csv(args.output_dir / "matched_control_coverage.csv", index=False)
    np.save(args.output_dir / "condition_mlp_validation_predictions.npy", mlp_prediction[validation_rows])
    np.save(
        args.output_dir / "condition_mlp_validation_sample_ids.npy",
        metadata.index.to_numpy(dtype=str)[validation_rows],
        allow_pickle=False,
    )

    summary = {
        "n_train_samples": int(train_rows.sum()),
        "n_validation_samples": int(validation_rows.sum()),
        "n_proteins": int(y_log2.shape[1]),
        "feature_encoder": encoder.to_dict(),
        "mlp": mlp_run_info,
        "matched_control_public_definition_warning": (
            "The tutorial table uses fewer samples than its prose definition. "
            "These results implement the disclosed exact-match fields without forcing official counts."
        ),
        "excluded_as_treatment_targets": sorted(EXCLUDED_TREATMENTS),
    }
    (args.output_dir / "stage1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(mlp_run_info, ensure_ascii=False, indent=2))
    print("\nAll treatment metrics:\n", all_frame.to_string(index=False))
    print("\nMatched subset metrics:\n", matched_frame.to_string(index=False))
    print("\nMatched control coverage:\n", coverage_frame.to_string(index=False))


if __name__ == "__main__":
    main()
