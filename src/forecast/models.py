"""Point-forecast model wrappers with a uniform forecaster interface.

Every wrapper exposes ``.fit(X, y) -> self``, ``.predict(X) -> Series`` (indexed
like ``X``), and ``.name`` so it drops into the rolling-origin harness exactly
like a baseline. Preprocessing (impute → robust scale) lives inside the wrapped
sklearn ``Pipeline`` so it is fit on train folds only, and the **training column
schema is enforced at predict** (a missing required column raises; unexpected
columns are dropped and reordered).
"""
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .constants import HORIZONS_H


class SklearnForecaster:
    """Wrap a scikit-learn regressor as a schema-enforcing forecaster."""

    def __init__(self, estimator, name: str, *, robust_scale: bool = True,
                 impute: bool = True):
        self.estimator = estimator
        self.name = name
        self.robust_scale = robust_scale
        self.impute = impute
        self.columns_: list[str] | None = None
        self.pipe_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnForecaster":
        self.columns_ = list(X.columns)
        steps = []
        if self.impute:
            steps.append(("impute", SimpleImputer(strategy="median")))
        if self.robust_scale:
            steps.append(("scale", RobustScaler()))
        steps.append(("est", self.estimator))
        self.pipe_ = Pipeline(steps)
        self.pipe_.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def _enforce_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"{self.name}: missing required columns {missing[:5]}...")
        return X[self.columns_]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = self._enforce_schema(X)
        yhat = self.pipe_.predict(X.to_numpy(dtype=float))
        return pd.Series(yhat, index=X.index, name=self.name)


def RidgeForecaster(alpha: float = 10.0, name: str = "ridge") -> SklearnForecaster:
    return SklearnForecaster(Ridge(alpha=alpha), name)


def LassoForecaster(alpha: float = 1e-3, name: str = "lasso") -> SklearnForecaster:
    return SklearnForecaster(Lasso(alpha=alpha, max_iter=5000), name)


def ElasticNetForecaster(alpha: float = 1e-3, l1_ratio: float = 0.5,
                         name: str = "elasticnet") -> SklearnForecaster:
    return SklearnForecaster(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000), name)


def HGBForecaster(
    learning_rate: float = 0.05, max_iter: int = 400, max_depth: int | None = None,
    max_leaf_nodes: int = 31, l2_regularization: float = 0.0, min_samples_leaf: int = 100,
    random_state: int = 0, name: str = "hgb",
) -> SklearnForecaster:
    """HistGradientBoosting — handles NaN natively, so no impute/scale."""
    est = HistGradientBoostingRegressor(
        learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes, l2_regularization=l2_regularization,
        min_samples_leaf=min_samples_leaf, early_stopping=True, random_state=random_state,
    )
    return SklearnForecaster(est, name, robust_scale=False, impute=False)


class DirectMultiHorizon:
    """DIRECT multi-horizon orchestrator: one fitted forecaster per horizon.

    ``factory(horizon_h, seed)`` returns a fresh forecaster. ``fit`` takes a dict
    of per-horizon ``(X, y)`` (each already aligned); ``predict`` takes a dict of
    per-horizon ``X`` and returns a dict of per-horizon prediction Series.
    """

    def __init__(self, factory, horizons: Sequence[int] = HORIZONS_H, seed: int = 0):
        self.factory = factory
        self.horizons = list(horizons)
        self.seed = seed
        self.models_: dict[int, object] = {}

    def fit(self, xy_by_h: dict[int, tuple[pd.DataFrame, pd.Series]]) -> "DirectMultiHorizon":
        for h in self.horizons:
            X, y = xy_by_h[h]
            self.models_[h] = self.factory(h, self.seed).fit(X, y)
        return self

    def predict(self, X_by_h: dict[int, pd.DataFrame]) -> dict[int, pd.Series]:
        return {h: self.models_[h].predict(X_by_h[h]) for h in self.horizons}
