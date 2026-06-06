"""Helpers for the feature-ablation sweep (notebooks/feature_ablation.py).

Two responsibilities:

1. ``build_engineered_design`` / ``build_seq_design`` — assemble the feature
   matrix for a *subset* of stations against pre-loaded sources already
   clipped to a fixed comparison window. Splitting this out of the notebook
   keeps the sweep loop tight and gives the report script a single canonical
   "what columns belong to station X" definition.

2. ``recommended_set`` — apply the 0.5%-RMSE selection rule to a long-form
   ablation result DataFrame so the report can both render and recommend
   from one place.

The station lists ``STATIONS_WAVE`` / ``STATIONS_WIND`` are derived from the
qld_ckan constants — the single source of truth for which CSVs the
downloader can produce.
"""
from dataclasses import dataclass

import pandas as pd

from qld_ckan.wave.constants import BUOYS
from qld_ckan.wind.constants import STATIONS as WIND_STATIONS

from . import features as feat

PRIMARY_BUOY = "mooloolaba"

# Every neighbour buoy in the network minus the primary. The primary is
# always present — it's the buoy whose hsig_m we're forecasting.
STATIONS_WAVE: list[str] = [s for s in BUOYS.keys() if s != PRIMARY_BUOY]

STATIONS_WIND: list[str] = list(WIND_STATIONS.keys())

ALL_STATIONS: list[str] = STATIONS_WAVE + STATIONS_WIND


def _split_stations(stations: list[str]) -> tuple[list[str], list[str]]:
    """Partition a flat station list into (wave_slugs, wind_slugs)."""
    wave = [s for s in stations if s in STATIONS_WAVE]
    wind = [s for s in stations if s in STATIONS_WIND]
    unknown = set(stations) - set(wave) - set(wind)
    if unknown:
        raise ValueError(f"Unknown station slugs: {sorted(unknown)}")
    return wave, wind


@dataclass
class PreloadedSources:
    """All sources loaded once and clipped to the all-station valid window.

    The window is computed so that EVERY station in the study has data for
    every row in ``wave.index`` — the latest-starting station (e.g. wide-bay
    2019) sets the start; the earliest-ending source sets the end. This is
    the single most important guarantee the ablation makes: adding or
    dropping a station changes only the column set seen by the model, not
    the training window.

    Subset views are produced cheaply by indexing — no re-loading and no
    re-overlapping.
    """
    wave: pd.DataFrame
    neighbours: dict[str, pd.Series]
    wind: pd.DataFrame
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    def subset(self, stations: list[str]) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame | None]:
        wave_slugs, wind_slugs = _split_stations(stations)
        nb = {s: self.neighbours[s] for s in wave_slugs}
        if wind_slugs:
            wind_cols = [
                c for c in self.wind.columns
                if any(c.startswith(f"{s}_") for s in wind_slugs)
            ]
            wind = self.wind[wind_cols]
        else:
            wind = None
        return self.wave, nb, wind


def compute_fixed_window(
    wave: pd.DataFrame,
    neighbours: dict[str, pd.Series],
    wind: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest start / latest end where every station has data.

    Differs from ``data.restrict_to_overlap`` in two ways:
      - Treats wave-neighbour series individually (not just the joint index).
      - Treats each wind station's columns as a group — late-arriving
        stations (e.g. southport from 2018) shrink the window even though
        ``wind.dropna(how="all")`` would not catch them.

    Both are needed so the ablation grid is apples-to-apples across subsets.
    """
    starts = [wave.dropna(how="all").index.min()]
    ends = [wave.dropna(how="all").index.max()]
    for s, series in neighbours.items():
        valid = series.dropna()
        if valid.empty:
            raise ValueError(f"Neighbour {s!r} has no non-NaN rows")
        starts.append(valid.index.min())
        ends.append(valid.index.max())
    for s in STATIONS_WIND:
        cols = [c for c in wind.columns if c.startswith(f"{s}_")]
        if not cols:
            continue
        # ``how="all"`` not ``"any"``: a single sparse column (e.g. lytton
        # wind_speed_std_ms only has 2021 data) must not collapse the window
        # — the Preprocessor will later drop columns whose NaN fraction is
        # too high.
        valid_idx = wind[cols].dropna(how="all").index
        if len(valid_idx) == 0:
            raise ValueError(f"Wind station {s!r} has no rows with any column populated")
        starts.append(valid_idx.min())
        ends.append(valid_idx.max())
    return max(starts), min(ends)


def load_all_sources(year_max: int | None = 2024) -> PreloadedSources:
    """Load primary buoy + every wave neighbour + every wind station, clip to fixed window.

    ``year_max`` mirrors the convention from ``notebooks/horizon_sweep.py``
    (cap at 2024 to keep wind coverage; 2025 is held out for real-world
    performance evaluation elsewhere).
    """
    from . import data

    wave = data.restrict_to_years(data.load_data(buoy=PRIMARY_BUOY), None, year_max)
    neighbours = data.load_neighbours(wave.index, STATIONS_WAVE)
    wind = data.load_wind(wave.index, STATIONS_WIND)
    if wind is None:
        raise RuntimeError("load_wind returned None for the full station list — check CSVs.")
    start, end = compute_fixed_window(wave, neighbours, wind)
    wave_clipped = wave.loc[start:end]
    nb_clipped = {k: v.loc[start:end] for k, v in neighbours.items()}
    wind_clipped = wind.loc[start:end]
    return PreloadedSources(
        wave=wave_clipped,
        neighbours=nb_clipped,
        wind=wind_clipped,
        window_start=start,
        window_end=end,
    )


def build_engineered_design(
    sources: PreloadedSources,
    stations: list[str],
    config: feat.FeatureConfig | None = None,
) -> pd.DataFrame:
    """Lag/rolling/momentum feature matrix for Ridge / HGB.

    ``stations`` is a flat list of wave-neighbour and/or wind-station slugs
    to include in addition to the always-present primary buoy. An empty
    list yields the primary-only baseline.

    Mirrors the build path used by ``notebooks/horizon_sweep.py::build_combo``
    so ablation runs slot into the same numeric space as the existing
    horizon-sweep entries.
    """
    wave, nb, wind = sources.subset(stations)
    merged, neighbour_cols, _ = feat.assemble_inputs(wave, nb, wind)
    primary_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
    X = feat.build_buoy_features(primary_only, config=config)
    if neighbour_cols:
        X = feat.add_neighbour_features(X, merged, neighbour_cols, config=config)
    if wind is not None:
        wind_cols = [c for c in wind.columns if not c.endswith("_deg")]
        X = feat.add_neighbour_features(X, wind, wind_cols, config=config)
    return X


def build_seq_design(
    sources: PreloadedSources,
    stations: list[str],
) -> pd.DataFrame:
    """Raw + circular + time frame for sequence models (GRU).

    No lag/rolling — the sequence model windows its own input. Matches the
    "raw" branch of ``notebooks/seq_playground.py::build_features``.
    """
    wave, nb, wind = sources.subset(stations)
    merged, _, _ = feat.assemble_inputs(wave, nb, wind)
    X = feat.build_seq_features(merged)
    if wind is not None:
        for col in wind.columns:
            X[col] = wind[col]
    return X


def recommended_set(
    report_df: pd.DataFrame,
    threshold: float = 0.005,
) -> dict[tuple[str, int], list[str]]:
    """Apply the keep-if-helps-or-hurts rule per (family, horizon).

    A station is kept when EITHER:
      - add-one RMSE improves the primary-only baseline by ≥ ``threshold``
        (relative), OR
      - drop-one RMSE worsens the ceiling by ≥ ``threshold`` (relative).

    Args:
        report_df: long-form rows with columns ``family``, ``horizon_h``,
            ``direction`` (one of ``baseline``/``ceiling``/``add``/``drop``),
            ``station`` (None for baseline/ceiling), ``RMSE``.
        threshold: relative-RMSE threshold (default 0.5%).

    Returns:
        dict keyed by (family, horizon_h) → list of station slugs to keep
        (sorted alphabetically for stable display).
    """
    out: dict[tuple[str, int], list[str]] = {}
    for (family, h), grp in report_df.groupby(["family", "horizon_h"]):
        base = grp[grp["direction"] == "baseline"]["RMSE"]
        ceil = grp[grp["direction"] == "ceiling"]["RMSE"]
        if base.empty or ceil.empty:
            continue
        base_rmse = float(base.iloc[0])
        ceil_rmse = float(ceil.iloc[0])
        keep: list[str] = []
        adds = grp[grp["direction"] == "add"].set_index("station")["RMSE"]
        drops = grp[grp["direction"] == "drop"].set_index("station")["RMSE"]
        stations = sorted(set(adds.index) | set(drops.index))
        for s in stations:
            add_gain = (base_rmse - float(adds[s])) / base_rmse if s in adds else 0.0
            drop_cost = (float(drops[s]) - ceil_rmse) / ceil_rmse if s in drops else 0.0
            if add_gain >= threshold or drop_cost >= threshold:
                keep.append(s)
        out[(family, int(h))] = keep
    return out
