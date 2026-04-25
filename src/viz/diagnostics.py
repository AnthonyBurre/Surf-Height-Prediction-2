"""Model-comparison and residual-analysis plots.

Accepts ``EvaluationResult`` objects directly (from ``forecast.evaluate``)
so plots stay in sync with the harness that produced them.
"""
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


class _ResultLike(Protocol):
    name: str
    metrics: dict[str, float]
    predictions: np.ndarray


def rmse_bar(
    results: list[_ResultLike],
    *,
    metric: str = "RMSE",
    title: str | None = None,
    ax: Axes | None = None,
    color: str = "#4c72b0",
) -> Axes:
    """Horizontal bar chart of a single metric across models, sorted low→high."""
    rows = [(r.name, r.metrics[metric]) for r in results]
    s = pd.Series({name: value for name, value in rows}).sort_values()

    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(s))))
    s.plot(kind="barh", ax=ax, color=color)
    ax.set(title=title or f"Test-set {metric}", xlabel=metric)
    ax.invert_yaxis()  # best model on top
    return ax


def residual_timeseries(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    *,
    model_name: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot residuals (y_true − y_pred) over time, with a zero reference line."""
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_true.index, name="pred")
    resid = (y_true - y_pred).dropna()

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    resid.plot(ax=ax, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    title = "Residuals over time"
    if model_name:
        title = f"{title} — {model_name}"
    ax.set(title=title, ylabel="y − ŷ")
    return ax


def residual_by_bin(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    *,
    bins: list[float] | None = None,
    statistic: str = "std",
    ylabel: str | None = None,
    title: str | None = None,
    color: str = "#55a868",
    ax: Axes | None = None,
) -> tuple[Axes, pd.DataFrame]:
    """Bin observations by value and plot a chosen residual statistic per bin.

    ``statistic`` is a pandas aggregation name (``std``, ``mean``, ``count``,
    ``median``). The returned DataFrame carries ``mean``, ``std``, ``count``
    for every bin regardless of which one was plotted.
    """
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_true.index, name="pred")
    resid = (y_true - y_pred).dropna()
    truth = y_true.loc[resid.index]

    if bins is None:
        bins = [
            float(truth.min()),
            *np.quantile(truth, [0.25, 0.5, 0.75, 0.9]).tolist(),
            float(truth.max()),
        ]
    bucketed = pd.cut(truth, bins=bins, include_lowest=True)
    grouped = resid.groupby(bucketed, observed=True).agg(["mean", "std", "count"])

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    grouped[statistic].plot(kind="bar", ax=ax, color=color, legend=False)
    ax.set(
        title=title or f"Residual {statistic} by bin",
        xlabel=truth.name or "observed value",
        ylabel=ylabel or f"{statistic} of errors",
    )
    plt.setp(ax.get_xticklabels(), rotation=0)
    return ax, grouped
