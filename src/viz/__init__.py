"""Plotting and exploratory visualisation — source-agnostic.

The module is organised by *what you are looking at*, not by what you
are looking at it with:

- ``timeseries``   — series in time; single source or overlay of many.
- ``correlation``  — correlation structure across features, lags, horizons, sources.
- ``diagnostics``  — model-comparison and residual-analysis plots.

Every function accepts an optional ``ax`` so panels can be composed into
a larger ``plt.subplots`` grid. For multi-source work (buoy comparisons,
overlaying atmospheric channels on buoy channels, etc.) the convention
is a ``dict[str, pd.Series]`` or ``dict[str, pd.DataFrame]`` keyed by
source label — that label ends up in the legend / row heading.
"""
from .correlation import (
    cross_source_correlation,
    feature_horizon_heatmap,
    lookback_horizon_heatmap,
)
from .diagnostics import residual_by_bin, residual_timeseries, rmse_bar
from .timeseries import autocorrelation_curve, plot_multi_source, plot_series

__all__ = [
    # correlation
    "cross_source_correlation",
    "feature_horizon_heatmap",
    "lookback_horizon_heatmap",
    # diagnostics
    "residual_by_bin",
    "residual_timeseries",
    "rmse_bar",
    # timeseries
    "autocorrelation_curve",
    "plot_multi_source",
    "plot_series",
]
