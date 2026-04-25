"""Feature engineering building blocks for wave-height forecasting.

All functions are pure — they take a DataFrame and return a new one —
so they compose in any order without surprising side effects.

Leakage rule: these helpers only look at times ``<= t`` for the row
indexed at ``t``. Pair with ``data.make_target`` (which shifts by
``-horizon``) to keep the forecast origin at ``t``.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CIRCULAR_COLS


@dataclass
class FeatureConfig:
    """Canonical knobs for lag/rolling/momentum feature construction.

    A single instance shared across an experiment guarantees that every
    model in that run sees the same feature set, making results directly
    comparable. The defaults reproduce the configuration used in
    forecast_v2 and the multi-buoy experiments.
    """
    # Primary buoy lags/rolling/momentum (in 30-min steps)
    lag_steps: list[int]    = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 96, 144])
    roll_windows: list[int] = field(default_factory=lambda: [12, 24, 48, 96])
    delta_steps: list[int]  = field(default_factory=lambda: [6, 12, 24, 48])
    # Neighbour-column lags/rolling (applied by add_neighbour_features)
    neighbour_lag_steps: list[int]    = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    neighbour_roll_windows: list[int] = field(default_factory=lambda: [6, 12, 24])


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

    The shift-by-one is a convention, not strictly a leakage fix at the
    current 12h horizon: ``hsig_m[t]`` is a legitimate feature at forecast
    time ``t``, so an unshifted rolling window over ``[t-w+1, t]`` doesn't
    expose the future label ``hsig_m[t+12h]``. We shift anyway for three
    reasons:

    1. **Robustness to horizon changes.** If ``HORIZON_STEPS`` is ever set
       to 0 (nowcasting), an unshifted window that includes ``t`` would
       leak the label directly. Shifting keeps the module safe by default.
    2. **Symmetry with ``add_lag_features``**, which only ever looks at
       strictly-past values. Treating rolling stats as "summary of the
       lags" makes the whole feature family interpretable as past-only.
    3. **Less collinearity with the raw column.** An unshifted ``roll4_mean``
       is ``0.25·hsig_m[t] + 0.75·(past)`` — nearly redundant with the raw
       feature already in the frame. The shifted version is independent.
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


def add_momentum(
    df: pd.DataFrame,
    columns: list[str],
    deltas: list[int],
) -> pd.DataFrame:
    """Append ``{col}_delta_{d} = col(t) - col(t-d)`` for each (col, d).

    Captures the rate of change / trend direction over multiple timescales.
    ``d`` is in 30-minute steps.
    """
    out = df.copy()
    for col in columns:
        for d in deltas:
            out[f"{col}_delta_{d}"] = df[col] - df[col].shift(d)
    return out


def build_mooloolaba_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Full primary-buoy feature matrix: circular + time + lags + rolling + momentum.

    ``df`` should contain only the primary buoy's columns (no neighbour
    columns). Call ``add_neighbour_features`` on the result to append
    cross-buoy inputs.
    """
    if config is None:
        config = FeatureConfig()
    return (
        df.pipe(encode_circular)
          .pipe(add_time_features)
          .pipe(add_lag_features,
                columns=["hsig_m", "hmax_m", "tp_s", "tz_s"],
                lags=config.lag_steps)
          .pipe(add_rolling_features,
                columns=["hsig_m", "tp_s", "hmax_m"],
                windows=config.roll_windows,
                stats=("mean", "std", "min", "max"))
          .pipe(add_momentum,
                columns=["hsig_m", "tp_s", "hmax_m"],
                deltas=config.delta_steps)
    )


def add_neighbour_features(
    X: pd.DataFrame,
    source_df: pd.DataFrame,
    columns: list[str],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Append raw + lag + rolling features for each neighbour column.

    Args:
        X: existing feature matrix to extend (typically from
            ``build_mooloolaba_features``).
        source_df: DataFrame containing the raw neighbour columns, indexed
            on the same DatetimeIndex as ``X``.
        columns: column names in ``source_df`` to use as neighbour inputs.
        config: feature configuration; defaults to ``FeatureConfig()``.

    The raw value at t (lag 0) plus ``config.neighbour_lag_steps`` shifted
    copies and ``config.neighbour_roll_windows`` rolling mean/std are added
    for each column.
    """
    if config is None:
        config = FeatureConfig()
    out = X.copy()
    for col in columns:
        out[col] = source_df[col]
        for lag in config.neighbour_lag_steps:
            out[f"{col}_lag{lag}"] = source_df[col].shift(lag)
        for w in config.neighbour_roll_windows:
            r = source_df[col].shift(1).rolling(window=w, min_periods=max(1, w // 2))
            out[f"{col}_roll{w}_mean"] = r.mean()
            out[f"{col}_roll{w}_std"]  = r.std()
    return out


def build_seq_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal input frame for sequence models: circular encoding + time features.

    No lag or rolling columns — the sequence model windows its own input
    over ``seq_len`` steps and is expected to learn temporal structure
    itself. Pass the result directly to an ``LSTMForecaster`` (or GRU/TCN).
    """
    return df.pipe(encode_circular).pipe(add_time_features)
