from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .constants import CONTROL_NAMES, EXCLUDED_TREATMENTS, MATCH_KEYS


def protein_mean(y_log2: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(np.asarray(y_log2)[train_rows], axis=0)
    if not np.isfinite(mean).all():
        missing = np.flatnonzero(~np.isfinite(mean))[:10].tolist()
        raise ValueError(f"Training protein mean is non-finite at columns {missing}")
    return mean.astype(np.float32)


def _row_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row[field] for field in MATCH_KEYS)


def matched_control_predictions(
    metadata: pd.DataFrame,
    y_log2: np.ndarray,
    candidate_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row positions, predictions and number of controls per target.

    This follows the public prose definition. The tutorial's published sample
    counts imply an additional undisclosed solvent-pairing or QC rule, so the
    caller must report the resulting coverage rather than forcing table counts.
    """
    if len(metadata) != len(y_log2):
        raise ValueError("metadata/y_log2 row mismatch")

    is_control = metadata["perturbation_no_concentration"].isin(CONTROL_NAMES).to_numpy()
    grouped_positions: dict[tuple[object, ...], list[int]] = {}
    for position in np.flatnonzero(is_control):
        grouped_positions.setdefault(_row_key(metadata.iloc[position]), []).append(int(position))

    lookup: dict[tuple[object, ...], tuple[np.ndarray, int]] = {}
    for key, positions in grouped_positions.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(np.asarray(y_log2)[positions], axis=0)
        lookup[key] = (mean.astype(np.float32), len(positions))

    matched_rows: list[int] = []
    predictions: list[np.ndarray] = []
    control_counts: list[int] = []
    for position in np.flatnonzero(candidate_rows):
        entry = lookup.get(_row_key(metadata.iloc[position]))
        if entry is None:
            continue
        prediction, count = entry
        if not np.isfinite(prediction).any():
            continue
        matched_rows.append(int(position))
        predictions.append(prediction)
        control_counts.append(count)

    if not matched_rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, y_log2.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.asarray(matched_rows, dtype=np.int64),
        np.stack(predictions),
        np.asarray(control_counts, dtype=np.int64),
    )


def treatment_rows(metadata: pd.DataFrame, split: str) -> np.ndarray:
    return (
        metadata["split_final"].eq(split)
        & ~metadata["perturbation_no_concentration"].isin(EXCLUDED_TREATMENTS)
    ).to_numpy()

