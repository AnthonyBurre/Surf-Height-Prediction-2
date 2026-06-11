"""Shared fixtures. Most tests run on small synthetic series so they never read
the multi-MB CSVs in ``data/``."""
import numpy as np
import pandas as pd
import pytest

from forecast.constants import SOURCE_TZ


@pytest.fixture
def synthetic_series() -> pd.Series:
    """A deterministic 30-min series: diurnal + annual cycles + AR(1) noise,
    on a tz-aware Brisbane grid, with two injected NaN gaps."""
    n = 48 * 400  # ~400 days
    idx = pd.date_range("2018-01-01", periods=n, freq="30min", tz=SOURCE_TZ)
    rng = np.random.default_rng(0)
    t = np.arange(n)
    diurnal = 0.5 * np.sin(2 * np.pi * t / 48)
    annual = 0.8 * np.sin(2 * np.pi * t / (48 * 365))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.1)
    y = pd.Series(1.5 + diurnal + annual + noise, index=idx, name="hsig_m")
    y.iloc[1000:1010] = np.nan          # short gap
    y.iloc[5000:5100] = np.nan          # longer gap
    return y


@pytest.fixture
def synthetic_frame(synthetic_series) -> pd.DataFrame:
    """A small wave-like observation frame derived from the synthetic series."""
    y = synthetic_series
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "hsig_m": y,
        "hmax_m": y * 1.6 + rng.normal(0, 0.05, len(y)),
        "tz_s": 6 + 0.5 * np.sin(2 * np.pi * np.arange(len(y)) / 48),
        "peak_dir_deg": (90 + 30 * np.sin(2 * np.pi * np.arange(len(y)) / 96)) % 360,
    }, index=y.index)


@pytest.fixture
def tmp_log(tmp_path):
    return tmp_path / "experiments.jsonl"
