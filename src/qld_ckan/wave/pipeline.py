import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import unify_frames
from .constants import BUOYS, COLUMN_RENAME_MAP, RESOURCE_IDS, SENTINEL_VALUE
from .downloader import fetch_all

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parents[3] / "data"

# Source timestamps are naive AEST (Queensland is fixed UTC+10, no DST), so
# localising to Australia/Brisbane attaches the correct offset before we
# convert to UTC for storage. Everything downstream — CSV, modelling, viz —
# sees UTC; multi-source joins (BOM/GFS reanalysis grids are UTC-native)
# stay trivial.
_SOURCE_TZ = "Australia/Brisbane"
_SAMPLING_FREQ = "30min"


def unify(resource_ids: dict[int, str] | None = None) -> pd.DataFrame:
    """Download all years and concatenate into a single DataFrame.

    Columns are already standardised per-year by the downloader; no further
    cleaning has been applied. Pass the result to ``clean()``.
    """
    return unify_frames(fetch_all(resource_ids))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, drop rows with invalid timestamps, set a sorted,
    tz-aware UTC DatetimeIndex on a regular 30-minute grid, coerce measurement
    columns to numeric, and replace the -99.9 sentinel with NaN.

    Gaps in the source data become NaN rows on the reindexed grid so that
    downstream lag/rolling features have a well-defined temporal axis.
    """
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df = df.dropna(subset=["datetime_utc"])
    df = df.set_index("datetime_utc")
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
    df.index.name = "datetime_utc"
    df.index = df.index.tz_localize(_SOURCE_TZ).tz_convert("UTC")
    return df


def run(
    buoy: str = "mooloolaba",
    output_path: str | Path | None = None,
    resource_ids: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Full pipeline: download → clean → save CSV.

    Args:
        buoy: key into ``constants.BUOYS`` (e.g. ``"brisbane"``); determines
            both the resource IDs and the default output filename.
        output_path: destination for the unified CSV; defaults to
            ``data/{buoy}_wave_data_{first_year}-{last_year}.csv``.
        resource_ids: explicit year → CKAN resource ID mapping; overrides
            the ``buoy`` lookup when provided.

    Returns:
        The cleaned DataFrame (tz-aware DatetimeIndex, standardised columns).
    """
    if resource_ids is None:
        resource_ids = BUOYS[buoy]
    if output_path is None:
        years = sorted(resource_ids)
        output_path = _DATA_DIR / f"{buoy}_wave_data_{years[0]}-{years[-1]}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = clean(unify(resource_ids))
    df.to_csv(output_path)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df
