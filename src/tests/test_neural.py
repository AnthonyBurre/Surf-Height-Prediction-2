import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # skipped without the 'forecast' extra

from forecast import neural  # noqa: E402
from forecast.constants import SOURCE_TZ  # noqa: E402
from forecast.windows import make_windows, windows_for_index  # noqa: E402


def _frame(n=4000):
    idx = pd.date_range("2018-01-01", periods=n, freq="30min", tz=SOURCE_TZ)
    t = np.arange(n)
    y = 1.5 + 0.5 * np.sin(2 * np.pi * t / 48) + 0.05 * np.random.default_rng(0).normal(size=n)
    return pd.DataFrame({"hsig_m": y, "tz_s": 6.0}, index=idx)


def test_make_windows_shapes_and_origin_alignment():
    df = _frame()
    X, y, idx = make_windows(df, ["hsig_m", "tz_s"], context_len=48, horizon_h=12,
                             target_col="hsig_m")
    assert X.shape[1:] == (48, 2)
    assert X.shape[0] == len(y) == len(idx)
    # window ends at the origin: last channel-0 value == hsig at origin
    assert np.isclose(X[0, -1, 0], df["hsig_m"].reindex([idx[0]]).iloc[0], atol=1e-5)


def test_windows_for_index_drops_nan_windows():
    df = _frame()
    df.iloc[100:110, 0] = np.nan
    origins = df.index[200:400]
    X, kept = windows_for_index(df, ["hsig_m"], context_len=48, origins=origins)
    assert len(kept) == len(X)
    assert len(kept) <= len(origins)


def test_seq_forecaster_is_deterministic_per_seed():
    df = _frame(3000)
    y = df["hsig_m"].shift(-24).rename("y")  # h=12 target
    X = pd.DataFrame(index=df.index[:-50])
    yv = y.reindex(X.index)
    Xa = X.loc[yv.dropna().index]
    ya = yv.dropna()

    def make(seed):
        return neural.SeqForecaster(df, ["hsig_m", "tz_s"], context_len=48, horizon_h=12,
                                    arch="gru", hidden=16, layers=1, epochs=3, seed=seed)

    p1 = make(0).fit(Xa, ya).predict(Xa)
    p2 = make(0).fit(Xa, ya).predict(Xa)
    assert np.allclose(p1.dropna().to_numpy(), p2.dropna().to_numpy())


def test_context_for_horizon_grows():
    assert neural.context_for_horizon(6) < neural.context_for_horizon(72)
