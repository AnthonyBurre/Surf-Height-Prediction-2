"""Plotting and exploratory visualisation — source-agnostic.

Organised by *when in the pipeline* you reach for the chart:

- ``timeseries``   — shared primitives for any time-indexed series.
                     Used both post-download (raw buoy / wind channels,
                     multi-source overlays, autocorrelation) and post-
                     modeling (prediction traces, residual series).
- ``eda``          — post-download exploratory plots that only make
                     sense *before* a model exists: feature-horizon
                     screening and cross-source correlation.
- ``diagnostics``  — post-modeling plots driven by ``EvaluationResult``
                     objects from ``forecast.evaluate``: model-comparison
                     bars and residual analysis.

Every function accepts an optional ``ax`` so panels can be composed into
a larger ``plt.subplots`` grid. For multi-source work (buoy comparisons,
overlaying atmospheric channels on buoy channels, etc.) the convention
is a ``dict[str, pd.Series]`` or ``dict[str, pd.DataFrame]`` keyed by
source label — that label ends up in the legend / row heading.
"""
from .diagnostics import residual_by_bin, residual_timeseries, rmse_bar
from .eda import cross_source_correlation, feature_horizon_heatmap
from .timeseries import autocorrelation_curve, plot_multi_source, plot_series

__all__ = [
    # timeseries (shared primitives)
    "autocorrelation_curve",
    "plot_multi_source",
    "plot_series",
    # eda (post-download)
    "cross_source_correlation",
    "feature_horizon_heatmap",
    # diagnostics (post-modeling)
    "residual_by_bin",
    "residual_timeseries",
    "rmse_bar",
]
