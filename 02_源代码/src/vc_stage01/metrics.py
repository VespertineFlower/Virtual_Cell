from __future__ import annotations

import numpy as np


def masked_regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    observation_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: truth={truth.shape}, prediction={prediction.shape}")
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if observation_mask is not None:
        observation_mask = np.asarray(observation_mask, dtype=bool)
        if observation_mask.shape != truth.shape:
            raise ValueError("observation_mask shape mismatch")
        valid &= observation_mask
    if not valid.any():
        raise ValueError("No valid truth/prediction pairs")

    error = prediction - truth
    squared_error = np.where(valid, error**2, 0.0)
    rmse = np.sqrt(squared_error.sum() / valid.sum())

    observed_truth = truth[valid]
    global_mean = observed_truth.mean()
    global_denom = ((observed_truth - global_mean) ** 2).sum()
    global_r2 = np.nan if global_denom <= 0 else 1.0 - squared_error.sum() / global_denom

    counts = valid.sum(axis=0)
    truth_sum = np.where(valid, truth, 0.0).sum(axis=0)
    truth_mean = np.divide(
        truth_sum,
        counts,
        out=np.full(truth.shape[1], np.nan),
        where=counts > 0,
    )
    centered = np.where(valid, truth - truth_mean, 0.0)
    protein_denom = (centered**2).sum(axis=0)
    protein_numer = squared_error.sum(axis=0)
    protein_r2 = 1.0 - np.divide(
        protein_numer,
        protein_denom,
        out=np.full(truth.shape[1], np.nan),
        where=protein_denom > 0,
    )

    return {
        "samples": int(valid.any(axis=1).sum()),
        "valid_values": int(valid.sum()),
        "log2_rmse": float(rmse),
        "global_r2": float(global_r2),
        "protein_r2_median": float(np.nanmedian(protein_r2)),
        "proteins_with_r2": int(np.isfinite(protein_r2).sum()),
    }

