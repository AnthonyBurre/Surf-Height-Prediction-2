"""Significant wave height (``hsig_m``) forecasting across 6-72h horizons.

Top-level exports cover the common workflow: load data, build target,
split, engineer features, fit baselines or regressors, score, and log.
``make_target`` defaults to the 12h horizon, but every step accepts a
``horizon_steps`` override (see ``forecast.config``).

Sequence-model forecasters (``SimpleRNNForecaster``, ``GRUForecaster``,
``LSTMForecaster``, ``TCNForecaster``) plus ``auto_device`` require
``torch`` and are loaded lazily — ``import forecast`` stays light, and
the first reference to one of those names triggers the torch import
(with a clear ImportError if torch is not installed).

Example
-------
>>> from forecast import load_data, make_target, chronological_split
>>> from forecast import PersistenceForecaster, evaluate
>>> df = load_data(buoy="mooloolaba")
>>> y = make_target(df)
>>> X_train, X_test, y_train, y_test = chronological_split(df, y)
>>> evaluate(PersistenceForecaster(), X_train, y_train, X_test, y_test)
"""
from .baselines import (
    ClimatologyHourForecaster,
    PersistenceForecaster,
)
from .config import (
    CIRCULAR_COLS,
    FEATURE_COLS,
    HORIZON_HOURS,
    HORIZON_STEPS,
    SAMPLING_FREQ_MINUTES,
    TARGET_COL,
    hours_to_steps,
)
from .data import (
    SOURCE_TZ,
    chronological_split,
    load_data,
    load_neighbours,
    load_wind,
    make_target,
    restrict_to_overlap,
    restrict_to_years,
)
from .evaluate import (
    EvaluationResult,
    compare,
    evaluate,
)
from .preprocess import (
    Preprocessor,
    drop_sparse_columns,
    mean_impute,
    scale_features,
)
from .experiments import (
    best_metric,
    best_run,
    compose_run_name,
    evaluate_and_log,
    find_runs,
    latest_metric,
    latest_run,
    log_run,
    read_log,
    recent_runs,
    wind_tag,
)
from .features import (
    FeatureConfig,
    add_lag_features,
    add_momentum,
    add_neighbour_features,
    add_rolling_features,
    assemble_inputs,
    build_buoy_features,
    build_seq_features,
    encode_circular,
)
from .metrics import bias, mae, rmse, skill_score, summarise

# Names defined in .neural that should be lazy-loaded so plain
# ``import forecast`` doesn't drag torch in. ``auto_device`` belongs
# here for the same reason — it inspects the live torch runtime.
_NEURAL_NAMES = frozenset({
    "GRUForecaster",
    "LSTMForecaster",
    "SimpleRNNForecaster",
    "TCNForecaster",
    "auto_device",
})


def __getattr__(name: str):
    if name in _NEURAL_NAMES:
        from . import neural
        return getattr(neural, name)
    raise AttributeError(f"module 'forecast' has no attribute {name!r}")


__all__ = [
    # config
    "CIRCULAR_COLS",
    "FEATURE_COLS",
    "HORIZON_HOURS",
    "HORIZON_STEPS",
    "SAMPLING_FREQ_MINUTES",
    "TARGET_COL",
    "hours_to_steps",
    # data
    "SOURCE_TZ",
    "chronological_split",
    "load_data",
    "load_neighbours",
    "load_wind",
    "make_target",
    "restrict_to_overlap",
    "restrict_to_years",
    # features
    "FeatureConfig",
    "add_lag_features",
    "add_momentum",
    "add_neighbour_features",
    "add_rolling_features",
    "assemble_inputs",
    "build_buoy_features",
    "build_seq_features",
    "encode_circular",
    # baselines
    "ClimatologyHourForecaster",
    "PersistenceForecaster",
    # neural (lazy)
    "GRUForecaster",
    "LSTMForecaster",
    "SimpleRNNForecaster",
    "TCNForecaster",
    "auto_device",
    # metrics
    "bias",
    "mae",
    "rmse",
    "skill_score",
    "summarise",
    # evaluate
    "EvaluationResult",
    "compare",
    "evaluate",
    # preprocess
    "Preprocessor",
    "drop_sparse_columns",
    "mean_impute",
    "scale_features",
    # experiments
    "best_metric",
    "best_run",
    "compose_run_name",
    "evaluate_and_log",
    "find_runs",
    "latest_metric",
    "latest_run",
    "log_run",
    "read_log",
    "recent_runs",
    "wind_tag",
]
