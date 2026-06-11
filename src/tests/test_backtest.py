import numpy as np
import pandas as pd

from forecast.backtest import (
    RollingOriginResult, block_bootstrap_ci, paired_block_bootstrap, rolling_origin,
    suggest_block_len,
)
from forecast.constants import SOURCE_TZ
from forecast.splits import RollingOriginSplitter


class _ConstModel:
    """Predicts the per-fold training mean (a trivial but valid forecaster)."""
    name = "mean"

    def fit(self, X, y):
        self._m = float(y.mean())
        return self

    def predict(self, X):
        return pd.Series(self._m, index=X.index)


def _xy(n=12000):
    idx = pd.date_range("2018-01-01", periods=n, freq="30min", tz=SOURCE_TZ)
    rng = np.random.default_rng(0)
    y = pd.Series(np.cumsum(rng.normal(0, 0.1, n)) + 5, index=idx)
    X = pd.DataFrame({"f": y.shift(1).fillna(0)}, index=idx)
    return X, y


def test_rolling_origin_result_shape():
    X, y = _xy()
    spl = RollingOriginSplitter(n_folds=3, val_size=1000, embargo_steps=48)
    res = rolling_origin(lambda s: _ConstModel(), X, y, spl, metrics=("rmse", "mae"))
    assert isinstance(res, RollingOriginResult)
    assert list(res.per_fold.columns) == ["rmse", "mae"]
    assert len(res.per_fold) == 3
    assert {"y_true", "y_pred", "fold"}.issubset(res.predictions.columns)
    assert set(res.mean) == {"rmse", "mae"}


def test_block_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    resid = rng.normal(0, 1.0, 4000)
    ci = block_bootstrap_ci(resid, reducer="rmse", block_len=50, n_boot=400, seed=0)
    assert ci.lo < ci.point < ci.hi
    assert ci.se > 0


def test_paired_bootstrap_flags_real_and_null_differences():
    rng = np.random.default_rng(2)
    n = 4000
    resid_b = rng.normal(0, 1.0, n)
    # model A clearly better: half the error magnitude
    resid_a = resid_b * 0.5
    real = paired_block_bootstrap(resid_a, resid_b, loss="squared", block_len=50, n_boot=400)
    assert real.significant and real.delta < 0      # A better
    # identical errors -> no difference
    null = paired_block_bootstrap(resid_b, resid_b.copy(), block_len=50, n_boot=400)
    assert not null.significant


def test_suggest_block_len_grows_with_autocorrelation():
    rng = np.random.default_rng(3)
    white = rng.normal(0, 1, 3000)
    ar = np.zeros(3000)
    for i in range(1, 3000):
        ar[i] = 0.95 * ar[i - 1] + rng.normal(0, 0.1)
    assert suggest_block_len(ar) > suggest_block_len(white)
