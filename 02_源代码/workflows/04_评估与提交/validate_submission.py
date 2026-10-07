from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = ROOT / "data" / "WAYB_WAYC_metadata_test(1).csv"
DEFAULT_CONTRACT = ROOT / "resources" / "feature_contract.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a prediction file against the frozen contract.")
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    id_column = contract["id_column"]
    expected_rows = int(contract["test_rows"])
    expected_proteins = int(contract["protein_columns"])
    contract_columns_path = args.contract.with_suffix(".csv")
    expected_columns = None
    if contract_columns_path.exists():
        expected_columns = pd.read_csv(contract_columns_path, usecols=["protein"])["protein"].astype(str).tolist()
        if len(expected_columns) != expected_proteins:
            raise ValueError("feature_contract.csv count does not match feature_contract.json")

    metadata = pd.read_csv(args.metadata, usecols=[id_column])
    prediction = pd.read_csv(args.prediction)
    if id_column not in prediction.columns:
        raise ValueError(f"Missing required ID column: {id_column}")

    ids = prediction[id_column].astype(str)
    expected_ids = metadata[id_column].astype(str)
    values = prediction.drop(columns=[id_column]).to_numpy(dtype=np.float64, copy=False)
    checks = {
        "rows": len(prediction),
        "protein_columns": prediction.shape[1] - 1,
        "protein_columns_exact_order": (
            prediction.columns[1:].astype(str).tolist() == expected_columns
            if expected_columns is not None else None
        ),
        "sample_id_exact_order": ids.tolist() == expected_ids.tolist(),
        "duplicate_sample_ids": int(ids.duplicated().sum()),
        "na_count": int(np.isnan(values).sum()),
        "finite": bool(np.isfinite(values).all()),
        "prediction_scale": contract["prediction_scale"],
        "value_min": float(np.min(values)),
        "value_max": float(np.max(values)),
    }
    errors = []
    if checks["rows"] != expected_rows:
        errors.append(f"expected {expected_rows} rows")
    if checks["protein_columns"] != expected_proteins:
        errors.append(f"expected {expected_proteins} protein columns")
    if checks["protein_columns_exact_order"] is False:
        errors.append("protein columns or order differ from feature_contract.csv")
    if not checks["sample_id_exact_order"]:
        errors.append("sample_ID values or order differ from test metadata")
    if checks["duplicate_sample_ids"]:
        errors.append("duplicate sample_ID values found")
    if checks["na_count"] or not checks["finite"]:
        errors.append("prediction contains NA or infinite values")

    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("Submission validation failed: " + "; ".join(errors))
    print("Submission validation passed.")


if __name__ == "__main__":
    main()
