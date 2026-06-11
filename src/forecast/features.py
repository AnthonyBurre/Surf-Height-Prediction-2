"""Leakage-safe feature engineering for the engineered design matrix (Phase 6).

The matrix is origin-indexed: row `t` may use observations up to and including
`t` (the present is known at the origin), but **window statistics are shifted to
end at `t-1`** so a rolling stat never includes the current step, and **angular
variables are circular-encoded** (sin, cos) before any averaging/lagging so
359°→1° is 2° apart, not 358°.

No fitting happens here (no scaling/imputation) — that lives in the model
pipeline so it can be fit on train folds only.
"""
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import (
    TARGET_COL, WAVE_DIR_COLS, WIND_DIR_COLS,
)

# Defaults: lags/windows in 30-min steps, placed near autocorrelation knees
# (0.5h, 1h, 1.5h, 3h, 6h, 12h, 1d, 2d, 3d) — refined by EDA Fig 2.4.
DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 96, 144)
DEFAULT_WINDOWS = (6, 12, 48, 96)          # 3h, 6h, 1d, 2d
DEFAULT_DELTAS = (1, 6, 48)                # 0.5h, 3h, 1d momentum
DEFAULT_ROLL_STATS = ("mean", "std", "min", "max")

DIR_COLS_ALL = set(WAVE_DIR_COLS) | set(WIND_DIR_COLS)


def circular_encode(df: pd.DataFrame, dir_cols: Sequence[str]) -> pd.DataFrame:
    """Replace each angular degree column with ``{col}_sin``/``{col}_cos``."""
    out = df.drop(columns=[c for c in dir_cols if c in df.columns])
    for col in dir_cols:
        if col not in df.columns:
            continue
        rad = np.deg2rad(df[col].to_numpy(dtype=float))
        out[f"{col}_sin"] = np.sin(rad)
        out[f"{col}_cos"] = np.cos(rad)
    return out


def lag_features(df: pd.DataFrame, cols: Sequence[str], lags: Sequence[int]) -> pd.DataFrame:
    """``x(t-L)`` for each lag L (steps). L≥1 is strictly past; L=0 is the present."""
    return pd.concat(
        {f"{c}_lag{L}": df[c].shift(L) for c in cols for L in lags}, axis=1
    )


def rolling_features(
    df: pd.DataFrame, cols: Sequence[str], windows: Sequence[int],
    stats: Sequence[str] = DEFAULT_ROLL_STATS,
) -> pd.DataFrame:
    """Rolling stats **ending at `t-1`** (``.shift(1)`` — the leakage guard)."""
    out = {}
    for c in cols:
        s = df[c]
        for w in windows:
            roll = s.rolling(w, min_periods=max(2, w // 2))
            for stat in stats:
                out[f"{c}_roll{w}_{stat}"] = getattr(roll, stat)().shift(1)
    return pd.concat(out, axis=1)


def delta_features(df: pd.DataFrame, cols: Sequence[str], periods: Sequence[int]) -> pd.DataFrame:
    """Momentum ``x(t) - x(t-p)`` (past-only: both terms ≤ t)."""
    return pd.concat(
        {f"{c}_delta{p}": df[c].diff(p) for c in cols for p in periods}, axis=1
    )


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical hour-of-day / day-of-year / month encodings (sin, cos)."""
    hour = index.hour + index.minute / 60.0
    doy = index.dayofyear
    out = pd.DataFrame(index=index)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    value_cols: Sequence[str] | None = None,
    dir_cols: Sequence[str] | None = None,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    deltas: Sequence[int] = DEFAULT_DELTAS,
    roll_stats: Sequence[str] = DEFAULT_ROLL_STATS,
    add_calendar: bool = True,
    keep_current: bool = True,
) -> pd.DataFrame:
    """Assemble the engineered, origin-indexed design matrix.

    Angular columns are circular-encoded first, then everything is linear:
    current values (``keep_current``), lags, rolling stats (ending at `t-1`),
    momentum deltas, and cyclical calendar features. ``value_cols`` defaults to
    every numeric non-direction column; ``dir_cols`` defaults to the known
    direction columns present (raw or ``{source}__`` namespaced).
    """
    if dir_cols is None:
        dir_cols = [c for c in df.columns if c.split("__")[-1] in DIR_COLS_ALL]
    enc = circular_encode(df, dir_cols)

    if value_cols is None:
        value_cols = list(enc.columns)
    else:  # keep the sin/cos derived from requested dir cols
        derived = [c for c in enc.columns if c.endswith(("_sin", "_cos"))]
        value_cols = list(dict.fromkeys(list(value_cols) + derived))
    value_cols = [c for c in value_cols if c in enc.columns]

    parts = []
    if keep_current:
        parts.append(enc[value_cols].add_suffix("_now"))
    if lags:
        parts.append(lag_features(enc, value_cols, lags))
    if windows:
        parts.append(rolling_features(enc, value_cols, windows, roll_stats))
    if deltas:
        parts.append(delta_features(enc, value_cols, deltas))
    if add_calendar:
        parts.append(calendar_features(enc.index))

    X = pd.concat(parts, axis=1)
    X.index.name = "datetime"
    return X


def target_value_cols(df: pd.DataFrame) -> list[str]:
    """The target buoy's own columns (bare, un-namespaced) for a primary-only set."""
    return [c for c in df.columns if "__" not in c]
