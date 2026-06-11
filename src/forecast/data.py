"""Load the cleaned QLD CSVs and assemble model-ready observation frames.

The :mod:`qld_ckan` ETL writes one CSV per source to ``data/`` with names like
``mooloolaba_wave_data_2015-2025.csv``. Exact spans are not known a priori, so
everything here **globs** by source slug rather than hard-coding filenames.

All frames are tz-aware on the ``Australia/Brisbane`` (fixed UTC+10) axis. Wave
sources are native 30-minute; wind sources are native hourly and get
forward-filled onto the 30-minute grid by :func:`build_dataset` (matching the
README's description of the wind join).
"""
import glob
from pathlib import Path
from typing import Sequence

import pandas as pd

from .constants import (
    CADENCE, DATA_DIR, SOURCE_TZ, TARGET_BUOY, TARGET_COL, WAVE_COLS, WIND_COLS,
)


def _find_one(pattern: str, data_dir: Path) -> Path:
    """Return the single CSV matching ``pattern`` under ``data_dir``.

    If several match (overlapping spans), the widest-span file wins so callers
    get the most history. Raises if none match.
    """
    matches = sorted(Path(data_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern!r} under {data_dir}. "
            f"Run `python -m qld_ckan` to generate it."
        )
    if len(matches) == 1:
        return matches[0]
    # Pick the file whose year span (parsed from the trailing _{a}-{b}) is widest.
    def span(p: Path) -> int:
        stem = p.stem.rsplit("_", 1)[-1]
        try:
            a, b = stem.split("-")
            return int(b) - int(a)
        except ValueError:
            return -1
    return max(matches, key=span)


def _read_csv_tz(path: Path) -> pd.DataFrame:
    """Read one cleaned CSV with a tz-aware ``datetime`` index on the Brisbane axis."""
    df = pd.read_csv(path, index_col="datetime", parse_dates=["datetime"])
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(SOURCE_TZ)
    else:
        idx = idx.tz_convert(SOURCE_TZ)
    df.index = idx
    df.index.name = "datetime"
    return df.sort_index()


def _reindex_grid(df: pd.DataFrame, freq: str = CADENCE) -> pd.DataFrame:
    """Reindex onto a gap-free regular grid spanning the data (gaps -> NaN rows)."""
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
    return df.reindex(full).rename_axis("datetime")


def load_wave(buoy: str = TARGET_BUOY, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a wave-buoy CSV on the canonical 30-minute grid."""
    df = _read_csv_tz(_find_one(f"{buoy}_wave_data_*.csv", data_dir))
    df = df.reindex(columns=[c for c in WAVE_COLS if c in df.columns])
    return _reindex_grid(df, CADENCE)


def load_wind(station: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a wind-station CSV on its native hourly grid."""
    df = _read_csv_tz(_find_one(f"{station}_wind_data_*.csv", data_dir))
    df = df.reindex(columns=[c for c in WIND_COLS if c in df.columns])
    return _reindex_grid(df, "1h")


def available_sources(data_dir: Path = DATA_DIR) -> dict[str, list[str]]:
    """Discover which wave buoys and wind stations are present in ``data_dir``."""
    def slugs(kind: str) -> list[str]:
        out = []
        for p in sorted(glob.glob(str(Path(data_dir) / f"*_{kind}_data_*.csv"))):
            out.append(Path(p).name.split(f"_{kind}_data_")[0])
        return out
    return {"wave": slugs("wave"), "wind": slugs("wind")}


def load_target(data_dir: Path = DATA_DIR) -> pd.Series:
    """The prediction target: Mooloolaba ``hsig_m`` on the 30-minute grid."""
    return load_wave(TARGET_BUOY, data_dir)[TARGET_COL].rename(TARGET_COL)


def build_dataset(
    buoys: Sequence[str] = (TARGET_BUOY,),
    stations: Sequence[str] = (),
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Join selected sources onto the target buoy's 30-minute grid.

    The first buoy is the target and keeps bare column names (``hsig_m`` …);
    every other source is namespaced ``{source}__{col}`` so columns never
    collide. Wind stations are reindexed onto the 30-minute grid by
    forward-fill. Gap rows are preserved as NaN. The index is the target
    buoy's full grid.
    """
    if not buoys:
        raise ValueError("build_dataset needs at least one buoy (the target).")
    target_buoy, *neighbours = buoys
    base = load_wave(target_buoy, data_dir)
    index = base.index
    frames = [base]  # target columns stay bare

    for buoy in neighbours:
        nb = load_wave(buoy, data_dir).reindex(index)
        frames.append(nb.add_prefix(f"{buoy}__"))

    for station in stations:
        w = load_wind(station, data_dir)
        # forward-fill hourly readings onto the 30-min grid, then align.
        w = w.reindex(w.index.union(index)).ffill().reindex(index)
        frames.append(w.add_prefix(f"{station}__"))

    out = pd.concat(frames, axis=1)
    out.index.name = "datetime"
    return out


def common_overlap(df: pd.DataFrame, cols: Sequence[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last index where every column in ``cols`` is non-NaN.

    Used by grouped source ablation so every variant trains on identical rows.
    """
    present = df[list(cols)].notna().all(axis=1)
    if not present.any():
        raise ValueError(f"No rows where all of {list(cols)} are present.")
    idx = df.index[present]
    return idx.min(), idx.max()
