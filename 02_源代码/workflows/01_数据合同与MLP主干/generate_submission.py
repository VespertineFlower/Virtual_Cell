from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vc_stage01.constants import METADATA_TEST  # noqa: E402
from vc_stage01.features import ConditionFeatureEncoder  # noqa: E402
from vc_stage01.mlp import load_checkpoint_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 condition-MLP test submission.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage1" / "condition_mlp.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage1" / "prediction.csv",
    )
    args = parser.parse_args()

    metadata = pd.read_csv(args.data_dir / METADATA_TEST).set_index("sample_ID")
    if not metadata.index.is_unique:
        raise ValueError("Test sample_ID must be unique")

    # Read encoder metadata without loading any test proteome truth.
    import torch

    checkpoint_metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = ConditionFeatureEncoder.from_dict(checkpoint_metadata["encoder"])
    features = encoder.transform(metadata)
    prediction, checkpoint = load_checkpoint_predictions(args.checkpoint, features)
    protein_names = [str(name) for name in checkpoint["protein_names"]]
    if prediction.shape != (len(metadata), len(protein_names)):
        raise ValueError("Prediction shape does not match test samples and feature contract")
    if not np.isfinite(prediction).all():
        raise ValueError("Submission contains NA or inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(prediction, index=metadata.index, columns=protein_names)
    submission.index.name = "sample_ID"
    submission.to_csv(args.output)

    # Reopen the written file and verify the external artifact, not only memory.
    reopened = pd.read_csv(args.output, index_col="sample_ID")
    checks = {
        "rows": len(reopened),
        "protein_columns": reopened.shape[1],
        "sample_id_exact_order": reopened.index.astype(str).tolist() == metadata.index.astype(str).tolist(),
        "protein_columns_exact_order": reopened.columns.tolist() == protein_names,
        "duplicate_sample_ids": int(reopened.index.duplicated().sum()),
        "na_count": int(reopened.isna().sum().sum()),
        "finite": bool(np.isfinite(reopened.to_numpy()).all()),
        "prediction_scale": "log2",
        "warning": (
            "This file follows the frozen 4,232-protein contract produced by the strict "
            "training missing-rate < 0.697 rule."
        ),
    }
    if not all(
        [
            checks["sample_id_exact_order"],
            checks["protein_columns_exact_order"],
            checks["duplicate_sample_ids"] == 0,
            checks["na_count"] == 0,
            checks["finite"],
        ]
    ):
        raise AssertionError(f"Submission verification failed: {checks}")
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"Submission written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
