from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baselines import treatment_rows
from .constants import EXCLUDED_TREATMENTS, MATCH_KEYS, VALIDATION_SPLITS
from .stage2 import strict_matched_controls


CONTEXT_KEYS = MATCH_KEYS
MODULE_WEIGHTS = {
    "module1_fc": 0.25,
    "module2_fidelity": 0.20,
    "module3_context_residual": 0.20,
    "module4_drug_residual": 0.20,
    "module5_both_time": 0.10,
    "module6_dep": 0.05,
}


@dataclass(frozen=True)
class ProxyReference:
    context_delta_mean: dict[tuple[object, ...], np.ndarray]
    drug_delta_mean: dict[str, np.ndarray]
    train_fc_pairs: int
    context_groups: int
    drug_groups: int

    def summary(self) -> dict[str, int]:
        return {
            "train_fc_pairs": self.train_fc_pairs,
            "context_groups": self.context_groups,
            "drug_groups": self.drug_groups,
        }


def _key(metadata: pd.DataFrame, position: int, fields: tuple[str, ...]) -> tuple[object, ...]:
    row = metadata.iloc[position]
    return tuple(row[field] for field in fields)


def _nanmean_rows(values: np.ndarray, positions: list[int]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values[np.asarray(positions)], axis=0).astype(np.float32)


def fit_proxy_reference(metadata: pd.DataFrame, y_log2: np.ndarray,
                        train_rows: np.ndarray) -> ProxyReference:
    """Fit official-score proxy references without reading validation truth."""
    train_treatments = train_rows & ~metadata["perturbation_no_concentration"].isin(
        EXCLUDED_TREATMENTS).to_numpy()
    rows, controls, _ = strict_matched_controls(
        metadata, y_log2, train_treatments, train_rows)
    truth = np.asarray(y_log2)[rows]
    delta = np.where(np.isfinite(truth) & np.isfinite(controls), truth - controls, np.nan)

    context_positions: dict[tuple[object, ...], list[int]] = {}
    drug_positions: dict[str, list[int]] = {}
    for local, position in enumerate(rows):
        context_positions.setdefault(_key(metadata, int(position), CONTEXT_KEYS), []).append(local)
        drug = str(metadata.iloc[int(position)]["perturbation_no_concentration"])
        drug_positions.setdefault(drug, []).append(local)
    context_means = {key: _nanmean_rows(delta, positions)
                     for key, positions in context_positions.items()}
    drug_means = {drug: _nanmean_rows(delta, positions)
                  for drug, positions in drug_positions.items()}
    return ProxyReference(context_means, drug_means, len(rows),
                          len(context_means), len(drug_means))


def _pearson(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> float:
    selected = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    if selected.sum() < 2:
        return float("nan")
    xv = np.asarray(x, dtype=np.float64)[selected]
    yv = np.asarray(y, dtype=np.float64)[selected]
    xv -= xv.mean()
    yv -= yv.mean()
    denominator = np.sqrt(np.square(xv).sum() * np.square(yv).sum())
    return float(np.dot(xv, yv) / denominator) if denominator > 1e-12 else float("nan")


def _axis_corr_r2(prediction: np.ndarray, truth: np.ndarray,
                  valid: np.ndarray, axis: int) -> tuple[float, float, int]:
    if axis == 0:
        prediction, truth, valid = prediction.T, truth.T, valid.T
    correlations = []
    r2_values = []
    for pred_row, truth_row, mask_row in zip(prediction, truth, valid):
        selected = mask_row & np.isfinite(pred_row) & np.isfinite(truth_row)
        if selected.sum() < 2:
            continue
        p = pred_row[selected].astype(np.float64)
        t = truth_row[selected].astype(np.float64)
        pc = p - p.mean()
        tc = t - t.mean()
        denominator = np.sqrt(np.square(pc).sum() * np.square(tc).sum())
        if denominator > 1e-12:
            correlations.append(float(np.dot(pc, tc) / denominator))
        tss = np.square(tc).sum()
        if tss > 1e-12:
            r2_values.append(float(1.0 - np.square(p - t).sum() / tss))
    return (float(np.mean(correlations)) if correlations else float("nan"),
            float(np.mean(r2_values)) if r2_values else float("nan"),
            len(r2_values))


def _corr_score(value: float) -> float:
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(value) else float("nan")


def _r2_score(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else float("nan")


def _mean_finite(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positives)


def _dep_metrics(delta_prediction: np.ndarray, delta_truth: np.ndarray,
                 valid: np.ndarray, threshold: float = 1.0) -> dict[str, float]:
    direction, high_corr, recall_at_k, f1_values, auprc = [], [], [], [], []
    for pred_row, truth_row, mask_row in zip(delta_prediction, delta_truth, valid):
        selected = mask_row & np.isfinite(pred_row) & np.isfinite(truth_row)
        if selected.sum() < 2:
            continue
        p = pred_row[selected].astype(np.float64)
        t = truth_row[selected].astype(np.float64)
        labels = np.abs(t) > threshold
        n_positive = int(labels.sum())
        if n_positive == 0:
            continue
        predicted_positive = np.abs(p) > threshold
        true_positive = int((labels & predicted_positive).sum())
        precision = true_positive / max(int(predicted_positive.sum()), 1)
        recall = true_positive / n_positive
        f1_values.append(2 * precision * recall / max(precision + recall, 1e-12))
        direction.append(float(np.mean(np.sign(p[labels]) == np.sign(t[labels]))))
        high_corr.append(_pearson(p[labels], t[labels], np.ones(n_positive, dtype=bool)))
        top_k = np.argpartition(np.abs(p), -n_positive)[-n_positive:]
        recall_at_k.append(float(labels[top_k].sum() / n_positive))
        auprc.append(_average_precision(np.abs(p), labels))
    raw_corr = _mean_finite(high_corr)
    components = {
        "dep_direction_accuracy": _mean_finite(direction),
        "dep_high_effect_pcc": raw_corr,
        "dep_recall_at_k": _mean_finite(recall_at_k),
        "dep_f1": _mean_finite(f1_values),
        "dep_auprc": _mean_finite(auprc),
    }
    components["module6_score"] = _mean_finite([
        components["dep_direction_accuracy"], _corr_score(raw_corr),
        components["dep_recall_at_k"], components["dep_f1"],
        components["dep_auprc"],
    ])
    return components


def score_proxy_split(metadata: pd.DataFrame, y_log2: np.ndarray,
                      full_prediction: np.ndarray, split: str,
                      train_rows: np.ndarray,
                      reference: ProxyReference) -> dict[str, Any]:
    targets = treatment_rows(metadata, split)
    # The official evaluator owns true matched controls for held-out entities.
    # Using them here is evaluation-only: models and train-fitted residual
    # references never receive these values.
    evaluation_control_pool = np.ones(len(metadata), dtype=bool)
    rows, controls, control_counts = strict_matched_controls(
        metadata, y_log2, targets, evaluation_control_pool)
    truth = np.asarray(y_log2)[rows]
    prediction = np.asarray(full_prediction)[rows]
    absolute_valid = np.isfinite(truth) & np.isfinite(prediction)
    delta_valid = absolute_valid & np.isfinite(controls)
    delta_truth = truth - controls
    delta_prediction = prediction - controls

    fc_pcc = _pearson(delta_prediction, delta_truth, delta_valid)
    sample_pcc, sample_r2, sample_n = _axis_corr_r2(
        prediction, truth, absolute_valid, axis=1)
    protein_pcc, protein_r2, protein_n = _axis_corr_r2(
        prediction, truth, absolute_valid, axis=0)
    fidelity = _mean_finite([
        _corr_score(sample_pcc), _r2_score(sample_r2),
        _corr_score(protein_pcc), _r2_score(protein_r2),
    ])

    context_mean = np.full_like(delta_truth, np.nan, dtype=np.float32)
    drug_mean = np.full_like(delta_truth, np.nan, dtype=np.float32)
    context_matched = 0
    drug_matched = 0
    for local, position in enumerate(rows):
        context = reference.context_delta_mean.get(_key(metadata, int(position), CONTEXT_KEYS))
        if context is not None:
            context_mean[local] = context
            context_matched += 1
        drug = str(metadata.iloc[int(position)]["perturbation_no_concentration"])
        drug_reference = reference.drug_delta_mean.get(drug)
        if drug_reference is not None:
            drug_mean[local] = drug_reference
            drug_matched += 1
    context_valid = delta_valid & np.isfinite(context_mean)
    drug_valid = delta_valid & np.isfinite(drug_mean)
    context_pcc = _pearson(delta_prediction - context_mean,
                           delta_truth - context_mean, context_valid)
    drug_pcc = _pearson(delta_prediction - drug_mean,
                        delta_truth - drug_mean, drug_valid)
    dep = _dep_metrics(delta_prediction, delta_truth, delta_valid)

    record: dict[str, Any] = {
        "split": split,
        "target_treatments": int(targets.sum()),
        "matched_control_samples": int(len(rows)),
        "matched_control_fraction": float(len(rows) / max(targets.sum(), 1)),
        "mean_controls_per_sample": float(control_counts.mean()) if len(rows) else 0.0,
        "absolute_valid_values": int(absolute_valid.sum()),
        "delta_valid_values": int(delta_valid.sum()),
        "context_reference_samples": context_matched,
        "drug_reference_samples": drug_matched,
        "fc_pcc": fc_pcc,
        "module1_score": _corr_score(fc_pcc),
        "sample_pcc_mean": sample_pcc,
        "sample_r2_mean": sample_r2,
        "sample_vectors_scored": sample_n,
        "protein_pcc_mean": protein_pcc,
        "protein_r2_mean": protein_r2,
        "protein_vectors_scored": protein_n,
        "module2_score": fidelity,
        "context_residual_pcc": context_pcc,
        "module3_score": _corr_score(context_pcc),
        "drug_residual_pcc": drug_pcc,
        "module4_score": _corr_score(drug_pcc),
    }
    record.update(dep)
    record["module5_score"] = _mean_finite([record["module1_score"], fidelity])
    return record


def aggregate_proxy_score(split_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {record["split"]: record for record in split_records}
    all_records = [by_split[split] for split in VALIDATION_SPLITS if split in by_split]
    module_values = {
        "module1_fc": _mean_finite([record["module1_score"] for record in all_records]),
        "module2_fidelity": _mean_finite([record["module2_score"] for record in all_records]),
        "module3_context_residual": by_split.get("val_chem_only", {}).get("module3_score", np.nan),
        "module4_drug_residual": by_split.get("val_strain_only", {}).get("module4_score", np.nan),
        "module5_both_time": _mean_finite([
            by_split.get(split, {}).get("module5_score", np.nan)
            for split in ("val_both", "val_time")]),
        "module6_dep": _mean_finite([record["module6_score"] for record in all_records]),
    }
    available_weight = sum(MODULE_WEIGHTS[name] for name, value in module_values.items()
                           if np.isfinite(value))
    weighted = sum(MODULE_WEIGHTS[name] * value for name, value in module_values.items()
                   if np.isfinite(value))
    return {
        **module_values,
        "proxy_score": float(weighted / max(available_weight, 1e-12)),
        "available_weight": float(available_weight),
        "module_weights": MODULE_WEIGHTS,
        "normalization": "PCC mapped to [0,1]; R2 clipped to [0,1]; module composites averaged",
    }


def score_official_proxy(metadata: pd.DataFrame, y_log2: np.ndarray,
                         full_prediction: np.ndarray, train_rows: np.ndarray,
                         reference: ProxyReference | None = None
                         ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = reference or fit_proxy_reference(metadata, y_log2, train_rows)
    split_records = [score_proxy_split(
        metadata, y_log2, full_prediction, split, train_rows, reference)
        for split in VALIDATION_SPLITS]
    return split_records, aggregate_proxy_score(split_records)
