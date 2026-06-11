"""Source-agnostic plotting for this project.

EDA figures (Phase 2) and results figures (Phase 11) operate on plain pandas
objects and ``forecast`` result records — never on project globals — so they are
reusable across datasets. Import the style first::

    import viz
    viz.apply_style()
    fig = viz.coverage_matrix(frame)
    viz.save(fig, "notebooks/figures/wave_coverage.png")
"""
from ._style import apply_style, save
from .eda import (
    acf_pacf, coverage_matrix, decomposition, feature_horizon_screen,
    lead_lag_matrix, seasonality_calendar, target_distribution,
)
from .results import (
    forest_plot, importance_horizon, residual_diagnostics, skill_vs_horizon,
)

__all__ = [
    "apply_style", "save",
    "coverage_matrix", "target_distribution", "decomposition", "acf_pacf",
    "seasonality_calendar", "lead_lag_matrix", "feature_horizon_screen",
    "skill_vs_horizon", "forest_plot", "residual_diagnostics", "importance_horizon",
]
