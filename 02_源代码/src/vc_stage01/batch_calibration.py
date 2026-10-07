from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONTROL_NAMES, EXCLUDED_TREATMENTS


BATCH_KEYS = ("data_source", "instrument", "Yeast_cell_plate")


@dataclass(frozen=True)
class BatchCalibrator:
    corrections: dict[tuple[object, ...], np.ndarray]
    mode: str
    shrinkage: float
    scale: float
    oof_objective: float
    oof_baseline_objective: float
    batch_keys: tuple[str, ...] = BATCH_KEYS

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "shrinkage": self.shrinkage,
            "scale": self.scale,
            "oof_objective": self.oof_objective,
            "oof_baseline_objective": self.oof_baseline_objective,
            "oof_gain": self.oof_baseline_objective - self.oof_objective,
            "groups": len(self.corrections),
            "batch_keys": list(self.batch_keys),
            "fit_scope": "train control/QC residuals from grouped OOF Huber predictions",
        }


def _row_keys(metadata: pd.DataFrame,
              fields: tuple[str, ...] = BATCH_KEYS) -> list[tuple[object, ...]]:
    return [tuple(value) for value in metadata.loc[:, fields].itertuples(index=False, name=None)]


def _huber_sum(error: np.ndarray, valid: np.ndarray, delta: float = 1.0) -> tuple[float, int]:
    absolute = np.abs(error)
    loss = np.where(absolute <= delta, 0.5 * error * error,
                    delta * (absolute - 0.5 * delta))
    return float(np.where(valid, loss, 0.0).sum(dtype=np.float64)), int(valid.sum())


def _pool_mask(metadata: pd.DataFrame, train_rows: np.ndarray, mode: str) -> np.ndarray:
    perturbation = metadata["perturbation_no_concentration"]
    controls = perturbation.isin(CONTROL_NAMES).to_numpy()
    qc = perturbation.eq("Quality Control").to_numpy()
    if mode == "controls":
        return train_rows & controls
    if mode == "controls_qc":
        return train_rows & (controls | qc)
    if mode == "qc":
        return train_rows & qc
    raise ValueError(f"Unsupported calibration pool: {mode}")


def _fit_group_statistics(metadata: pd.DataFrame, truth: np.ndarray,
                          prediction: np.ndarray, pool: np.ndarray
                          ) -> dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]]:
    keys = _row_keys(metadata)
    groups: dict[tuple[object, ...], list[int]] = {}
    for position in np.flatnonzero(pool):
        groups.setdefault(keys[position], []).append(int(position))
    statistics = {}
    for key, positions in groups.items():
        target = np.asarray(truth)[positions]
        pred = np.asarray(prediction)[positions]
        valid = np.isfinite(target) & np.isfinite(pred)
        residual = np.where(valid, target - pred, 0.0)
        statistics[key] = (
            residual.sum(axis=0, dtype=np.float64).astype(np.float32),
            valid.sum(axis=0).astype(np.float32),
        )
    return statistics


def _make_corrections(statistics: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]],
                      shrinkage: float, scale: float) -> dict[tuple[object, ...], np.ndarray]:
    corrections = {}
    for key, (sums, counts) in statistics.items():
        denominator = counts + shrinkage
        corrections[key] = np.divide(
            scale * sums, denominator,
            out=np.zeros_like(sums, dtype=np.float32), where=denominator > 0,
        ).astype(np.float32)
    return corrections


def _objective(metadata: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
               target_rows: np.ndarray,
               corrections: dict[tuple[object, ...], np.ndarray]) -> float:
    keys = _row_keys(metadata)
    numerator = 0.0
    denominator = 0
    by_key: dict[tuple[object, ...], list[int]] = {}
    for position in np.flatnonzero(target_rows):
        by_key.setdefault(keys[position], []).append(int(position))
    zeros = np.zeros(truth.shape[1], dtype=np.float32)
    for key, positions in by_key.items():
        target = np.asarray(truth)[positions]
        pred = np.asarray(prediction)[positions] + corrections.get(key, zeros)[None, :]
        valid = np.isfinite(target) & np.isfinite(pred)
        value, count = _huber_sum(pred - target, valid)
        numerator += value
        denominator += count
    return numerator / max(denominator, 1)


def fit_batch_calibrator(metadata: pd.DataFrame, y_log2: np.ndarray,
                         oof_prediction: np.ndarray, train_rows: np.ndarray,
                         modes: tuple[str, ...] = ("controls", "controls_qc", "qc"),
                         shrinkages: tuple[float, ...] = (0.0, 1.0, 4.0, 16.0, 64.0, 256.0),
                         scales: tuple[float, ...] = (0.25, 0.5, 1.0),
                         ) -> tuple[BatchCalibrator, list[dict[str, Any]]]:
    if np.asarray(oof_prediction).shape != np.asarray(y_log2).shape:
        raise ValueError("OOF prediction must align to the full train/validation matrix")
    target_rows = train_rows & ~metadata["perturbation_no_concentration"].isin(
        EXCLUDED_TREATMENTS).to_numpy()
    baseline = _objective(metadata, y_log2, oof_prediction, target_rows, {})
    records: list[dict[str, Any]] = [{
        "mode": "none", "shrinkage": np.nan, "scale": 0.0,
        "oof_objective": baseline, "groups": 0,
    }]
    best = BatchCalibrator({}, "none", float("inf"), 0.0, baseline, baseline)
    for mode in modes:
        pool = _pool_mask(metadata, train_rows, mode)
        statistics = _fit_group_statistics(metadata, y_log2, oof_prediction, pool)
        for shrinkage in shrinkages:
            for scale in scales:
                corrections = _make_corrections(statistics, shrinkage, scale)
                objective = _objective(metadata, y_log2, oof_prediction,
                                       target_rows, corrections)
                records.append({
                    "mode": mode, "shrinkage": shrinkage, "scale": scale,
                    "oof_objective": objective, "groups": len(corrections),
                    "pool_samples": int(pool.sum()),
                })
                if objective < best.oof_objective - 1e-10:
                    best = BatchCalibrator(corrections, mode, shrinkage, scale,
                                           objective, baseline)
    return best, records


def apply_batch_calibration(calibrator: BatchCalibrator, metadata: pd.DataFrame,
                            prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(prediction, dtype=np.float32).copy()
    correction = np.zeros_like(result)
    keys = _row_keys(metadata, calibrator.batch_keys)
    for position, key in enumerate(keys):
        value = calibrator.corrections.get(key)
        if value is not None:
            correction[position] = value
    result += correction
    return result, correction
