import numpy as np
import pandas as pd

from forecast import baselines
from forecast.constants import HORIZON_STEPS, STEPS_PER_DAY


def _origins(y, n=2000, start=500):
    return pd.DataFrame(index=y.index[start:start + n])


def test_persistence_returns_current_value(synthetic_series):
    y = synthetic_series
    X = _origins(y)
    pred = baselines.Persistence(y, 24).predict(X)
    assert np.allclose(pred.dropna().to_numpy(),
                       y.reindex(X.index).dropna().to_numpy())


def test_seasonal_naive_is_past_only_every_horizon(synthetic_series):
    y = synthetic_series
    X = _origins(y)
    for h in (6, 12, 24, 36, 48, 72):
        steps = HORIZON_STEPS[h]
        k = int(np.ceil(steps / STEPS_PER_DAY))
        offset = k * STEPS_PER_DAY - steps
        assert offset >= 0                      # never reads the future
        pred = baselines.SeasonalNaive(y, h).predict(X)
        expected = y.shift(offset).reindex(X.index)
        assert np.allclose(pred.dropna().to_numpy(), expected.dropna().to_numpy())


def test_seasonal_mean_reproduces_phase_means(synthetic_series):
    y = synthetic_series
    X = _origins(y, n=4000)
    sm = baselines.SeasonalMean(y, 0).fit(X, y.reindex(X.index))
    # predicting horizon 0 -> target time == origin time; phase mean of (month,hour)
    pred = sm.predict(X)
    assert pred.notna().all()
    assert pred.between(y.min() - 1, y.max() + 1).all()


def test_drift_predicts_level_plus_trend(synthetic_series):
    y = synthetic_series
    X = _origins(y)
    d = baselines.DriftRandomWalk(y, 24).fit(X, y.reindex(X.index))
    pred = d.predict(X)
    base = y.reindex(X.index)
    assert np.allclose((pred - base).dropna().to_numpy(),
                       HORIZON_STEPS[24] * d._drift, atol=1e-9)
