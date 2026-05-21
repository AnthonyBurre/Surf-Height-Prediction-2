"""Load the unified CSV, build the forecast target, split chronologically.

Also hosts the multi-source loaders (neighbour buoys, wind stations) that the
notebook playgrounds previously duplicated.
"""
from pathlib import Path

import pandas as pd

from .config import HORIZON_STEPS, TARGET_COL
from .features import encode_circular

# Resolve relative to the repo root, not the caller's cwd, so notebooks and
# scripts both find the file without path juggling.
_DATA_DIR = Path(__file__).parents[2] / "data"

# Source observations are naive AEST (Queensland is fixed UTC+10, no DST).
# pipeline.clean tags every CSV index with Australia/Brisbane, so the index is
# AEST end-to-end and ``df.index.year`` naturally returns the source-data
# year. SOURCE_TZ is exported for downstream consumers that need to assert
# the convention (or convert to another tz for a join).
SOURCE_TZ = "Australia/Brisbane"


def load_data(
    buoy: str = "mooloolaba",
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a unified wave buoy CSV with a tz-aware AEST DatetimeIndex.

    With no ``path``: globs ``data/{buoy}_wave_data_*.csv`` and picks the
    longest-range match (lexicographic last → e.g. ``..._2015-2025.csv``
    beats ``..._2015-2024.csv``). Pass ``path`` for a specific file.

    The pipeline writes Brisbane (UTC+10) offsets, which ``read_csv`` parses
    straight back to a tz-aware index — no relocalise needed.
    """
    if path is None:
        matches = sorted(_DATA_DIR.glob(f"{buoy}_wave_data_*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No wave CSV found for buoy={buoy!r} in {_DATA_DIR}. "
                f"Run `python -m qld_ckan wave --buoy {buoy}` to generate it."
            )
        path = matches[-1]
    return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")


def restrict_to_years(
    df: pd.DataFrame,
    year_min: int | None,
    year_max: int | None,
) -> pd.DataFrame:
    """Slice ``df`` to AEST years in ``[year_min, year_max]`` (inclusive on both ends).

    Returns ``df`` unchanged when both bounds are ``None``. The frame is
    expected to carry an AEST-tagged index (the project-wide convention
    set by ``qld_ckan.{wave,wind}.pipeline.clean``); ``df.index.year``
    therefore reads the source-data year directly.
    """
    if year_min is None and year_max is None:
        return df
    yr = df.index.year
    mask = pd.Series(True, index=df.index)
    if year_min is not None:
        mask &= yr >= year_min
    if year_max is not None:
        mask &= yr <= year_max
    return df.loc[mask]


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


def load_neighbours(
    target_index: pd.DatetimeIndex,
    neighbours: list[str],
) -> dict[str, pd.Series]:
    """Load each neighbour's hsig_m series, reindexed onto ``target_index``.

    For each slug, globs ``data/{slug}_wave_data_*.csv`` and picks the
    longest-range match (lexicographic last → e.g. ``..._2015-2025.csv``
    beats ``..._2015-2024.csv``), matching ``load_data``'s convention.
    Raises ``FileNotFoundError`` if no CSV exists for a requested slug.
    """
    out: dict[str, pd.Series] = {}
    for name in neighbours:
        matches = sorted(_DATA_DIR.glob(f"{name}_wave_data_*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No wave CSV found for neighbour={name!r} in {_DATA_DIR}. "
                f"Run `python -m qld_ckan wave --buoy {name}` to generate it."
            )
        nb = pd.read_csv(matches[-1], parse_dates=["datetime"], index_col="datetime")
        out[name] = nb["hsig_m"].reindex(target_index)
    return out


def load_wind(
    target_index: pd.DatetimeIndex,
    stations: list[str],
) -> pd.DataFrame | None:
    """Hourly wind for one or more stations, sin/cos-encoded, ffill'd to ``target_index``.

    Returns ``None`` if no stations are requested. Columns are namespaced by
    station slug (e.g. ``deception-bay_wind_speed_ms``) so multi-station
    loads keep them distinct. The wind grid is hourly while ``target_index``
    is typically 30-min: each 30-min slot inherits the most recent past
    hourly reading via forward-fill, which is strictly past-only.

    For each slug, globs ``data/{slug}_wind_data_*.csv`` and picks the
    longest-range match (lexicographic last). Raises ``FileNotFoundError``
    if no CSV exists for a requested slug.
    """
    if not stations:
        return None
    frames: list[pd.DataFrame] = []
    for s in stations:
        matches = sorted(_DATA_DIR.glob(f"{s}_wind_data_*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No wind CSV found for station={s!r} in {_DATA_DIR}. "
                f"Run `python -m qld_ckan wind --station {s}` to generate it."
            )
        w = pd.read_csv(matches[-1], parse_dates=["datetime"], index_col="datetime")
        w = encode_circular(w, periods={"wind_dir_deg": 360.0})
        w = w.add_prefix(f"{s}_")
        frames.append(w.reindex(target_index, method="ffill"))
    return pd.concat(frames, axis=1)


def restrict_to_overlap(
    wave: pd.DataFrame,
    neighbours: dict[str, pd.Series],
    wind: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame | None]:
    """Clip every source to the overlap window so feature rows are aligned.

    Wind window wins when present (it is currently the shortest source);
    otherwise fall back to the intersection of neighbour-buoy windows. With
    neither, returns the inputs untouched.
    """
    if wind is not None:
        valid = wind.dropna(how="all")
        start, end = valid.index.min(), valid.index.max()
    elif neighbours:
        starts = [s.dropna().index.min() for s in neighbours.values()]
        ends   = [s.dropna().index.max() for s in neighbours.values()]
        start, end = max(starts), min(ends)
    else:
        return wave, neighbours, wind
    wave = wave.loc[start:end]
    neighbours = {k: v.loc[start:end] for k, v in neighbours.items()}
    if wind is not None:
        wind = wind.loc[start:end]
    return wave, neighbours, wind
