"""Feature engineering building blocks for wave-height forecasting.

All functions are pure — they take a DataFrame and return a new one —
so they compose in any order without surprising side effects.

Leakage rule: these helpers only look at times ``<= t`` for the row
indexed at ``t``. Pair with ``data.make_target`` (which shifts by
``-horizon``) to keep the forecast origin at ``t``.
"""
import numpy as np
import pandas as pd

from .config import CIRCULAR_COLS


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Append ``{col}_lag_{k}`` for every ``(col, k)`` in ``columns × lags``.

    Lag ``k`` is in 30-minute steps (so ``k=2`` means 1 hour ago). A
    positive lag looks backward; the produced columns have NaN for the
    first ``k`` rows where history is unavailable.
    """
    out = df.copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    stats: tuple[str, ...] = ("mean", "std"),
) -> pd.DataFrame:
    """Append rolling-window aggregates, right-aligned and lagged by 1 step.

    The shift-by-one is critical: a naive rolling mean includes the current
    observation, which at prediction time is part of the input. Shifting
    ensures the window summarises the past only and matches how the feature
    behaves at inference.
    """
    out = df.copy()
    valid_stats = {"mean", "std", "min", "max", "median"}
    unknown = set(stats) - valid_stats
    if unknown:
        raise ValueError(f"Unknown stats: {sorted(unknown)}; valid: {sorted(valid_stats)}")

    for col in columns:
        for w in windows:
            r = df[col].shift(1).rolling(window=w, min_periods=max(1, w // 2))
            for stat in stats:
                out[f"{col}_roll{w}_{stat}"] = getattr(r, stat)()
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encodings of hour-of-day and day-of-year.

    Sin/cos pairs let a linear model represent "similar hours are close"
    without a one-hot blowup, and avoid the Dec-31 → Jan-1 cliff.
    """
    out = df.copy()
    hour = df.index.hour + df.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    doy = df.index.dayofyear.to_numpy(dtype=float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def encode_circular(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    period_deg: float = 360.0,
) -> pd.DataFrame:
    """Replace each circular column with its sin/cos pair.

    Applied to the source columns themselves, so any downstream lag/rolling
    features already see the encoded representation when this is called
    first in the pipeline.
    """
    if columns is None:
        columns = [c for c in CIRCULAR_COLS if c in df.columns]
    out = df.copy()
    for col in columns:
        rad = 2 * np.pi * out[col] / period_deg
        out[f"{col}_sin"] = np.sin(rad)
        out[f"{col}_cos"] = np.cos(rad)
        out = out.drop(columns=col)
    return out
