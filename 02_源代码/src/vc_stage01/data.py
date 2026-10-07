from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    METADATA_TEST,
    METADATA_TRAIN_VAL,
    PROTEOME_TEST,
    PROTEOME_TRAIN_VAL,
    VALIDATION_SPLITS,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    required = {
        "sample_ID",
        "split_final",
        "Strains",
        "perturbation_no_concentration",
        "Medium",
        "Temperature",
        "pert_time",
        "pert_time_unit",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if metadata["sample_ID"].isna().any() or metadata["sample_ID"].duplicated().any():
        raise ValueError("metadata sample_ID must be non-null and unique")
    return metadata.set_index("sample_ID")


def read_proteome(path: Path) -> pd.DataFrame:
    headers = pd.read_csv(path, nrows=0).columns.tolist()
    if not headers or headers[0] != "sample_ID":
        raise ValueError("The first proteome column must be sample_ID")
    dtypes: dict[str, Any] = {"sample_ID": str}
    dtypes.update({column: np.float32 for column in headers[1:]})
    proteome = pd.read_csv(path, dtype=dtypes).set_index("sample_ID")
    if not proteome.index.is_unique:
        raise ValueError("proteome sample_ID must be unique")
    return proteome


def align_frames(metadata: pd.DataFrame, proteome: pd.DataFrame) -> pd.DataFrame:
    metadata_ids = set(metadata.index)
    proteome_ids = set(proteome.index)
    if metadata_ids != proteome_ids:
        missing = sorted(metadata_ids - proteome_ids)[:10]
        extra = sorted(proteome_ids - metadata_ids)[:10]
        raise ValueError(f"sample_ID mismatch: missing={missing}, extra={extra}")
    return proteome.loc[metadata.index]


def compute_feature_contract(
    metadata: pd.DataFrame,
    proteome: pd.DataFrame,
    missing_threshold: float,
) -> tuple[pd.Index, pd.Series]:
    if not 0 < missing_threshold <= 1:
        raise ValueError("missing_threshold must be in (0, 1]")
    train_ids = metadata.index[metadata["split_final"].eq("train")]
    if len(train_ids) == 0:
        raise ValueError("No split_final=train samples were found")
    missing_rate = proteome.loc[train_ids].isna().mean(axis=0)
    kept = missing_rate.index[missing_rate < missing_threshold]
    if len(kept) == 0:
        raise ValueError("The feature filter removed every protein")
    return kept, missing_rate


def make_log2_and_mask(
    proteome: pd.DataFrame,
    protein_names: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    values = proteome.loc[:, protein_names].to_numpy(dtype=np.float32)
    finite = np.isfinite(values)
    if np.any(values[finite] <= 0):
        raise ValueError("Observed raw intensities must be strictly positive before log2")
    y_log2 = np.full(values.shape, np.nan, dtype=np.float32)
    np.log2(values, out=y_log2, where=finite)
    return y_log2, finite


def audit_inputs(data_dir: Path, missing_threshold: float) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "metadata_train_val": data_dir / METADATA_TRAIN_VAL,
        "metadata_test": data_dir / METADATA_TEST,
        "proteome_train_val": data_dir / PROTEOME_TRAIN_VAL,
        "proteome_test": data_dir / PROTEOME_TEST,
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")

    meta_tv = read_metadata(paths["metadata_train_val"])
    meta_test = read_metadata(paths["metadata_test"])
    proteome_tv = align_frames(meta_tv, read_proteome(paths["proteome_train_val"]))

    test_header = pd.read_csv(paths["proteome_test"], nrows=0).columns.tolist()
    test_ids = pd.read_csv(paths["proteome_test"], usecols=["sample_ID"])["sample_ID"].astype(str)
    if test_ids.duplicated().any() or set(test_ids) != set(meta_test.index):
        raise ValueError("Test metadata/proteome sample_ID contract failed")
    if test_header[1:] != proteome_tv.columns.tolist():
        raise ValueError("Train/validation and test protein columns differ")

    kept, missing_rate = compute_feature_contract(meta_tv, proteome_tv, missing_threshold)
    observed = proteome_tv.to_numpy(dtype=np.float32)
    finite = np.isfinite(observed)

    audit = {
        "metadata_train_val_shape": list(meta_tv.shape),
        "metadata_test_shape": list(meta_test.shape),
        "proteome_train_val_shape": list(proteome_tv.shape),
        "proteome_test_shape": [len(test_ids), len(test_header) - 1],
        "split_counts": meta_tv["split_final"].value_counts().astype(int).to_dict(),
        "test_split_counts": meta_test["split_final"].value_counts().astype(int).to_dict(),
        "validation_split_counts_expected_present": {
            split: int((meta_tv["split_final"] == split).sum()) for split in VALIDATION_SPLITS
        },
        "train_test_sample_id_overlap": int(len(set(meta_tv.index) & set(meta_test.index))),
        "raw_observed_min": float(observed[finite].min()),
        "raw_observed_max": float(observed[finite].max()),
        "raw_nonpositive_count": int((observed[finite] <= 0).sum()),
        "overall_missing_fraction": float((~finite).mean()),
        "missing_threshold": missing_threshold,
        "kept_proteins": int(len(kept)),
        "removed_proteins": int(len(missing_rate) - len(kept)),
        "train_missing_rate_min": float(missing_rate.min()),
        "train_missing_rate_median": float(missing_rate.median()),
        "train_missing_rate_max": float(missing_rate.max()),
        "feature_contract_status": (
            f"Current data retain {len(kept)} proteins at "
            f"training missing_rate < {missing_threshold:.3f}."
        ),
    }

    source_info = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    return {
        "metadata": meta_tv,
        "proteome": proteome_tv,
        "kept": kept,
        "missing_rate": missing_rate,
    }, {"audit": audit, "sources": source_info}


def save_stage0_cache(
    output_dir: Path,
    objects: dict[str, Any],
    report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: pd.DataFrame = objects["metadata"]
    proteome: pd.DataFrame = objects["proteome"]
    kept: pd.Index = objects["kept"]
    missing_rate: pd.Series = objects["missing_rate"]
    y_log2, mask = make_log2_and_mask(proteome, kept)

    np.save(output_dir / "y_log2.npy", y_log2, allow_pickle=False)
    np.save(output_dir / "mask.npy", mask, allow_pickle=False)
    np.save(output_dir / "sample_ids.npy", metadata.index.to_numpy(dtype=str), allow_pickle=False)
    metadata.reset_index().to_csv(output_dir / "metadata_aligned.csv", index=False)
    pd.DataFrame(
        {
            "protein": kept,
            "train_missing_rate": missing_rate.loc[kept].to_numpy(),
        }
    ).to_csv(output_dir / "feature_contract.csv", index=False)
    (output_dir / "stage0_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
