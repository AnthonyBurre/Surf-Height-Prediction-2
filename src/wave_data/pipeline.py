import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import COLUMN_RENAME_MAP, RESOURCE_IDS, SENTINEL_VALUE
from .downloader import fetch_all

logger = logging.getLogger(__name__)

# Default output path relative to the project root, regardless of cwd
_DEFAULT_OUTPUT = Path(__file__).parents[2] / "data" / "mooloolaba_wave_data_2015-2025.csv"

# Queensland does not observe DST, so localising a naive AEST timestamp to
# Australia/Brisbane is a safe, fixed UTC+10 shift.
_BUOY_TZ = "Australia/Brisbane"
_SAMPLING_FREQ = "30min"


def unify(resource_ids: dict[int, str] | None = None) -> pd.DataFrame:
    """Download all years and concatenate into a single DataFrame.

    Columns are already standardised per-year by the downloader; no further
    cleaning has been applied. Pass the result to ``clean()``.
    """
    frames = fetch_all(resource_ids)
    if not frames:
        raise ValueError("No data was downloaded; cannot build unified dataset.")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, drop rows with invalid timestamps, set a sorted,
    tz-aware DatetimeIndex on a regular 30-minute grid, coerce measurement
    columns to numeric, and replace the -99.9 sentinel with NaN.

    Gaps in the source data become NaN rows on the reindexed grid so that
    downstream lag/rolling features have a well-defined temporal axis.
    """
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df = df.dropna(subset=["datetime_aest"])
    df = df.set_index("datetime_aest")
    df = df.sort_index()
    # Yearly files occasionally overlap at boundaries; drop duplicate
    # timestamps so reindex has a unique axis to align against.
    df = df[~df.index.duplicated(keep="first")]
    # CKAN can return numeric fields as strings; coerce once here so
    # downstream consumers never need to think about dtypes.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(SENTINEL_VALUE, np.nan)
    # Reindex onto the full 30-minute grid so gaps surface as NaN rows
    # rather than being silently absent.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=_SAMPLING_FREQ)
    df = df.reindex(full_index)
    df.index.name = "datetime_aest"
    df.index = df.index.tz_localize(_BUOY_TZ)
    return df


def run(output_path: str | Path = _DEFAULT_OUTPUT) -> pd.DataFrame:
    """Full pipeline: download → clean → save CSV.

    Args:
        output_path: destination for the unified CSV.

    Returns:
        The cleaned DataFrame (tz-aware DatetimeIndex, standardised columns).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = clean(unify())
    df.to_csv(output_path)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df
