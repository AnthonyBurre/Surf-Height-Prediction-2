"""Fit/predict/score harness and a comparison helper.

The harness handles NaN rows so every model sees a consistent training
subset regardless of how many lag/rolling features it uses.
"""
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .metrics import summarise


class Forecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class EvaluationResult:
    name: str
    metrics: dict[str, float]
    predictions: np.ndarray  # aligned with X_test.index (NaN where input had NaN)
    model: Any


def _finite_row_mask(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Rows where every feature and the target are finite."""
    return (~X.isna().any(axis=1) & ~y.isna()).to_numpy()


def evaluate(
    model: Forecaster,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    name: str | None = None,
    baseline_preds: np.ndarray | None = None,
) -> EvaluationResult:
    """Fit on finite-feature rows, predict on finite-feature rows, pad back.

    Why this split:
      - Models that need no lags (e.g. Persistence) get every training row.
      - Models with 48-step lags lose the first 48 rows; that's fine —
        the mask drops them and everyone else still aligns on the index.
      - sklearn raises on NaN in X at predict time, so we mask there too
        and fill NaN into the skipped rows. The returned array is still
        length-aligned with X_test.index for metric computation.
    """
    train_mask = _finite_row_mask(X_train, y_train)
    model.fit(X_train.loc[train_mask], y_train.loc[train_mask])

    predict_mask = (~X_test.isna().any(axis=1)).to_numpy()
    preds = np.full(len(X_test), np.nan)
    if predict_mask.any():
        preds[predict_mask] = np.asarray(
            model.predict(X_test.loc[predict_mask]), dtype=float
        )
    metrics = summarise(y_test, preds, y_pred_baseline=baseline_preds)
    return EvaluationResult(
        name=name or type(model).__name__,
        metrics=metrics,
        predictions=preds,
        model=model,
    )


def compare(results: list[EvaluationResult]) -> pd.DataFrame:
    """Stack per-model metrics into a sorted DataFrame (lowest RMSE first)."""
    rows = [{"model": r.name, **r.metrics} for r in results]
    df = pd.DataFrame(rows).set_index("model")
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE")
    return df
