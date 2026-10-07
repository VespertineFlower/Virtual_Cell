from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import MLP_CATEGORICAL_FIELDS


@dataclass(frozen=True)
class AnchorFold:
    name: str
    train_rows: np.ndarray
    validation_rows: np.ndarray


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def grouped_condition_fold_ids(metadata: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Deterministically keep exact experimental conditions in the same fold."""
    fields = list(MLP_CATEGORICAL_FIELDS) + ["pert_time", "pert_time_unit"]
    result = np.empty(len(metadata), dtype=np.int64)
    for i, row in enumerate(metadata.loc[:, fields].itertuples(index=False, name=None)):
        result[i] = _stable_hash("\x1f".join(map(str, row))) % n_folds
    return result


def balanced_entity_fold_ids(values: np.ndarray, eligible_rows: np.ndarray,
                             n_folds: int) -> tuple[np.ndarray, dict[str, int]]:
    """Assign whole entities to deterministic, approximately sample-balanced folds."""
    values = np.asarray(values).astype(str)
    entities, counts = np.unique(values[eligible_rows], return_counts=True)
    if not 2 <= n_folds <= len(entities):
        raise ValueError(f"n_folds must be between 2 and {len(entities)}")
    ordered = sorted(zip(entities.tolist(), counts.tolist()),
                     key=lambda item: (-item[1], _stable_hash(item[0])))
    loads = [0] * n_folds
    assignment: dict[str, int] = {}
    for entity, count in ordered:
        fold = min(range(n_folds), key=lambda candidate: (loads[candidate], candidate))
        assignment[entity] = fold
        loads[fold] += count
    ids = np.full(len(values), -1, dtype=np.int64)
    for entity, fold in assignment.items():
        ids[eligible_rows & (values == entity)] = fold
    return ids, assignment


def make_anchor_folds(metadata: pd.DataFrame, train_rows: np.ndarray, strategy: str,
                      n_folds: int = 5, late_time: float | None = None,
                      compound_field: str = "perturbation_no_concentration",
                      strain_field: str = "Strains") -> tuple[list[AnchorFold], np.ndarray, dict[str, object]]:
    """Build leakage-auditable folds and the rows for which OOD anchors exist."""
    train_rows = np.asarray(train_rows, dtype=bool)
    folds: list[AnchorFold] = []
    audit: dict[str, object] = {"strategy": strategy}

    if strategy == "grouped":
        fold_ids = grouped_condition_fold_ids(metadata, n_folds)
        for fold in range(n_folds):
            valid = train_rows & (fold_ids == fold)
            folds.append(AnchorFold(f"fold_{fold}", train_rows & ~valid, valid))
        anchor_rows = train_rows.copy()
    elif strategy == "leave_one_strain_out":
        values = metadata[strain_field].astype(str).to_numpy()
        entities = sorted(np.unique(values[train_rows]).tolist())
        for entity in entities:
            valid = train_rows & (values == entity)
            folds.append(AnchorFold(f"strain_{entity}", train_rows & ~valid, valid))
        anchor_rows = train_rows.copy()
        audit["held_out_entities"] = entities
    elif strategy == "leave_compound_group_out":
        values = metadata[compound_field].astype(str).to_numpy()
        fold_ids, assignment = balanced_entity_fold_ids(values, train_rows, n_folds)
        for fold in range(n_folds):
            valid = train_rows & (fold_ids == fold)
            folds.append(AnchorFold(f"compound_group_{fold}", train_rows & ~valid, valid))
        anchor_rows = train_rows.copy()
        audit["entity_to_fold"] = assignment
    elif strategy == "late_time_holdout":
        times = metadata["pert_time"].astype(float).to_numpy()
        cutoff = float(np.max(times[train_rows])) if late_time is None else float(late_time)
        valid = train_rows & (times >= cutoff)
        fit = train_rows & (times < cutoff)
        folds.append(AnchorFold(f"time_ge_{cutoff:g}", fit, valid))
        anchor_rows = valid
        audit.update({"cutoff": cutoff, "fit_time_max": float(np.max(times[fit])),
                      "held_out_times": sorted(np.unique(times[valid]).tolist())})
    else:
        raise ValueError(f"Unknown anchor strategy: {strategy}")

    covered = np.zeros(len(metadata), dtype=np.int16)
    records: list[dict[str, object]] = []
    for fold in folds:
        if not fold.train_rows.any() or not fold.validation_rows.any():
            raise RuntimeError(f"Empty anchor split: {fold.name}")
        if np.any(fold.train_rows & fold.validation_rows):
            raise RuntimeError(f"Train/validation overlap in {fold.name}")
        covered += fold.validation_rows.astype(np.int16)
        records.append({"name": fold.name, "train_samples": int(fold.train_rows.sum()),
                        "validation_samples": int(fold.validation_rows.sum())})
    if not np.all(covered[anchor_rows] == 1) or np.any(covered[~anchor_rows] != 0):
        raise RuntimeError("Anchor folds do not cover the intended rows exactly once")
    audit.update({"anchor_samples": int(anchor_rows.sum()), "folds": records})
    return folds, anchor_rows, audit
