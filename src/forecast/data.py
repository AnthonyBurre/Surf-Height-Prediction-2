"""Load the unified CSV, build the forecast target, split chronologically."""
from pathlib import Path

import pandas as pd

from .config import HORIZON_STEPS, TARGET_COL

# Resolve relative to the repo root, not the caller's cwd, so notebooks and
# scripts both find the file without path juggling.
_DEFAULT_CSV = Path(__file__).parents[2] / "data" / "mooloolaba_wave_data_2015-2025.csv"


def load_data(path: str | Path = _DEFAULT_CSV) -> pd.DataFrame:
    """Load the unified wave buoy CSV with a tz-aware UTC DatetimeIndex.

    The pipeline writes UTC offsets, which ``read_csv(parse_dates=...)``
    parses straight back to a tz-aware UTC index — no relocalise needed.
    """
    return pd.read_csv(path, parse_dates=["datetime_utc"], index_col="datetime_utc")


def make_target(
    df: pd.DataFrame,
    horizon_steps: int = HORIZON_STEPS,
    target_col: str = TARGET_COL,
) -> pd.Series:
    """Return y where y.loc[t] is the value of target_col at time t + horizon.

    The series is indexed at the *forecast origin* (t), not the target time
    (t+h). This matches how forecasters are used in production: "given data
    up to now, what will hsig_m be 12 hours from now?"

    The last ``horizon_steps`` rows will be NaN — there is no future value
    to target. Callers must drop these (or mask during evaluation).
    """
    y = df[target_col].shift(-horizon_steps)
    y.name = f"{target_col}_plus_{horizon_steps}"
    return y


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split by index position, not by shuffling.

    Time series must be split chronologically: shuffling leaks future
    information into the training set via autocorrelation with neighbouring
    observations.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    split = int(len(X) * (1 - test_frac))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
