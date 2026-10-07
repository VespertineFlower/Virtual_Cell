from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import MLP_CATEGORICAL_FIELDS


@dataclass
class ConditionFeatureEncoder:
    vocabularies: dict[str, list[str]]
    max_train_time: float

    @classmethod
    def fit(cls, train_metadata: pd.DataFrame) -> "ConditionFeatureEncoder":
        vocabularies = {
            field: sorted(train_metadata[field].astype(str).unique().tolist())
            for field in MLP_CATEGORICAL_FIELDS
        }
        max_train_time = float(train_metadata["pert_time"].astype(float).max())
        if max_train_time <= 0:
            raise ValueError("Training perturbation time must be positive")
        return cls(vocabularies=vocabularies, max_train_time=max_train_time)

    @property
    def output_dim(self) -> int:
        # One-hot blocks and two continuous time features (linear and log1p).
        # Unseen categories map to an all-zero block so no untrained random
        # coefficient is activated at validation time.
        return sum(len(vocab) for vocab in self.vocabularies.values()) + 2

    def transform(self, metadata: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for field in MLP_CATEGORICAL_FIELDS:
            vocab = self.vocabularies[field]
            mapping = {value: index for index, value in enumerate(vocab)}
            values = metadata[field].astype(str).to_numpy()
            block = np.zeros((len(metadata), len(vocab)), dtype=np.float32)
            indices = np.fromiter((mapping.get(value, -1) for value in values), dtype=int)
            known = indices >= 0
            block[np.arange(len(metadata))[known], indices[known]] = 1.0
            blocks.append(block)

        time = metadata["pert_time"].astype(float).to_numpy(dtype=np.float32)
        linear = (time / self.max_train_time).reshape(-1, 1)
        log_scaled = (
            np.log1p(time) / np.log1p(self.max_train_time)
        ).astype(np.float32).reshape(-1, 1)
        blocks.extend([linear.astype(np.float32), log_scaled])
        return np.concatenate(blocks, axis=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "categorical_fields": list(MLP_CATEGORICAL_FIELDS),
            "vocabularies": self.vocabularies,
            "max_train_time": self.max_train_time,
            "output_dim": self.output_dim,
            "unknown_policy": "all-zero block for unseen categories",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ConditionFeatureEncoder":
        vocabularies = {
            str(field): [str(value) for value in values]
            for field, values in dict(payload["vocabularies"]).items()
        }
        return cls(
            vocabularies=vocabularies,
            max_train_time=float(payload["max_train_time"]),
        )
