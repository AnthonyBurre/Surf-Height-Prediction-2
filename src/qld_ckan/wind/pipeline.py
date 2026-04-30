import logging
from pathlib import Path

import pandas as pd

from .. import unify_frames
from .constants import COLUMN_RENAME_MAP, RESOURCE_IDS, STATIONS
from .downloader import fetch_all

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parents[3] / "data"

# Source timestamps are naive AEST (Queensland is fixed UTC+10, no DST), so
# localising to Australia/Brisbane attaches the correct offset before we
# convert to UTC for storage. This matches the wave-side convention so the
# two frames join on a shared UTC index without any timezone fiddling.
_SOURCE_TZ = "Australia/Brisbane"
_SAMPLING_FREQ = "1h"


def unify(resource_ids: dict[int, str] | None = None) -> pd.DataFrame:
    """Download all years and concatenate into a single DataFrame.

    Raw columns are preserved; pass the result to ``clean()`` to get the
    standardised, time-indexed frame.
    """
    return unify_frames(fetch_all(resource_ids))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Build a UTC-indexed hourly frame of the wind columns.

    Combines the separate ``Date`` and ``Time`` fields into one timestamp,
    drops any pollutant / temperature columns the source happens to carry,
    coerces wind values to numeric, deduplicates, and reindexes onto a
    gap-free hourly grid so downstream lag/rolling features have a
    well-defined temporal axis.
    """
    # Source Date format varies by year: most years use ISO ("2024-01-01T00:00:00")
    # but some (e.g. Mountain Creek and Deception Bay 2019) emit DD/MM/YYYY.
    # `format="mixed", dayfirst=True` parses both: ISO strings are unambiguous,
    # and DD/MM/YYYY is interpreted day-first as intended. Without dayfirst,
    # pandas defaults to month-first and silently coerces ~95% of a DD/MM year
    # to NaT.
    timestamps = pd.to_datetime(
        df["Date"].astype(str).str[:10] + " " + df["Time"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )
    df = df.assign(datetime_utc=timestamps).dropna(subset=["datetime_utc"])

    # Keep only the renameable wind columns; pollutant / temperature fields
    # are out of scope for this module and their presence varies by year.
    keep = [c for c in COLUMN_RENAME_MAP if c in df.columns]
    df = df[["datetime_utc", *keep]].rename(columns=COLUMN_RENAME_MAP)

    df = df.set_index("datetime_utc").sort_index()
    # Year-boundary overlaps occasionally produce duplicate timestamps; keep
    # the first so reindex has a unique axis to align against.
    df = df[~df.index.duplicated(keep="first")]
    # CKAN can return numeric fields as strings; coerce once here so
    # downstream consumers never need to think about dtypes.
    df = df.apply(pd.to_numeric, errors="coerce")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=_SAMPLING_FREQ)
    df = df.reindex(full_index)
    df.index.name = "datetime_utc"
    df.index = df.index.tz_localize(_SOURCE_TZ).tz_convert("UTC")
    return df


def run(
    station: str = "mountain-creek",
    output_path: str | Path | None = None,
    resource_ids: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Full pipeline: download → clean → save CSV.

    Args:
        station: key into ``constants.STATIONS`` (e.g. ``"mountain-creek"``);
            determines both the resource IDs and the default output filename.
        output_path: destination for the unified CSV; defaults to
            ``data/{station}_wind_data_{first_year}-{last_year}.csv``.
        resource_ids: explicit year → CKAN resource ID mapping; overrides
            the ``station`` lookup when provided.

    Returns:
        The cleaned DataFrame (tz-aware DatetimeIndex, standardised columns).
    """
    if resource_ids is None:
        resource_ids = STATIONS[station]
    if output_path is None:
        years = sorted(resource_ids)
        output_path = _DATA_DIR / f"{station}_wind_data_{years[0]}-{years[-1]}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = clean(unify(resource_ids))
    df.to_csv(output_path)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df
