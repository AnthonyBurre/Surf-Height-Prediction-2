"""Phase-3 baselines — the references that turn absolute error into *skill*.

Every baseline follows the same duck-typed forecaster protocol as the ML models
(``.fit(X, y) -> self``, ``.predict(X) -> Series`` indexed like ``X``, ``.name``)
so it drops straight into the rolling-origin harness and is scored on identical
folds. Baselines bind the full *observed* target series at construction and read
only values at or before the origin `t`, so they are leakage-safe by
construction — including a same-phase seasonal-naive that stays past-only at
**every** horizon (it steps back whole periods until the reference time is ≤ t).

Persistence is the project's primary skill denominator (conventional, and what
the README pins). The lower envelope of all baselines per horizon is the
"better baseline" backdrop for the money chart.
"""
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import HORIZON_STEPS, STEPS_PER_DAY, STEPS_PER_HOUR


class Persistence:
    """``ŷ(t+h) = y(t)`` — last observed value carried forward."""

    name = "persistence"

    def __init__(self, y_full: pd.Series, horizon_h: int):
        self.y = y_full
        self.h = horizon_h

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Persistence":
        return self  # nothing to fit

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.y.reindex(X.index).rename(self.name)


class SeasonalNaive:
    """``ŷ(t+h) = y`` at the most recent same-phase time ≤ t.

    For a daily period ``m`` (48 steps) the reference is ``y(t - offset)`` with
    ``offset = ceil(steps/m)*m - steps ≥ 0``, so it never reads the future even
    when ``h`` exceeds one period.
    """

    name = "seasonal_naive"

    def __init__(self, y_full: pd.Series, horizon_h: int, period: int = STEPS_PER_DAY):
        self.y = y_full
        self.h = horizon_h
        self.period = period

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        steps = HORIZON_STEPS[self.h]
        k = math.ceil(steps / self.period)
        offset = k * self.period - steps  # >= 0  ->  reads y(t - offset)
        return self.y.shift(offset).reindex(X.index).rename(self.name)


class SeasonalMean:
    """Climatology: the (month × hour-of-day) historical mean of the target.

    Horizon-independent in form but evaluated at the *target* time's phase, so it
    shifts with the horizon. Fit on the training origins only.
    """

    name = "seasonal_mean"

    def __init__(self, y_full: pd.Series, horizon_h: int):
        self.y = y_full
        self.h = horizon_h
        self._table: pd.Series | None = None
        self._global = float("nan")

    @staticmethod
    def _phase(idx: pd.DatetimeIndex) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([idx.month, idx.hour], names=["month", "hour"])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalMean":
        obs = self.y.reindex(X.index).dropna()
        self._global = float(obs.mean())
        self._table = obs.groupby(self._phase(obs.index)).mean()
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        target_times = X.index + pd.Timedelta(hours=self.h)
        keys = self._phase(target_times)
        vals = self._table.reindex(keys).to_numpy()
        vals = np.where(np.isnan(vals), self._global, vals)
        return pd.Series(vals, index=X.index, name=self.name)


class DriftRandomWalk:
    """Random walk + linear drift: ``ŷ(t+h) = y(t) + steps * drift``."""

    name = "drift"

    def __init__(self, y_full: pd.Series, horizon_h: int):
        self.y = y_full
        self.h = horizon_h
        self._drift = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DriftRandomWalk":
        d = self.y.reindex(X.index).diff().mean()
        self._drift = 0.0 if pd.isna(d) else float(d)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        steps = HORIZON_STEPS[self.h]
        return (self.y.reindex(X.index) + steps * self._drift).rename(self.name)


class Theta:
    """Theta method (statsmodels ``ThetaModel``) — a strong classical baseline.

    Refitting per origin at 30-min cadence over ~190k origins is infeasible, so
    Theta is meant for a *thinned* origin grid (e.g. one per day) in the
    baselines notebook, refit on the expanding window up to each origin. Kept out
    of the full-grid harness deliberately.
    """

    name = "theta"

    def __init__(self, y_full: pd.Series, horizon_h: int, period: int = STEPS_PER_DAY):
        self.y = y_full
        self.h = horizon_h
        self.period = period

    def forecast_origin(self, origin: pd.Timestamp) -> float:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        steps = HORIZON_STEPS[self.h]
        hist = self.y.loc[:origin].dropna()
        if len(hist) < 3 * self.period:
            return float(hist.iloc[-1]) if len(hist) else float("nan")
        hist = hist.reset_index(drop=True)  # ThetaModel wants a clean series
        model = ThetaModel(hist, period=self.period, deseasonalize=True).fit()
        return float(model.forecast(steps).iloc[-1])


def all_baselines(y_full: pd.Series, horizon_h: int) -> list:
    """The per-origin baselines scored on the full rolling-origin folds."""
    return [
        Persistence(y_full, horizon_h),
        SeasonalNaive(y_full, horizon_h),
        SeasonalMean(y_full, horizon_h),
        DriftRandomWalk(y_full, horizon_h),
    ]
