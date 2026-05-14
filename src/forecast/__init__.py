"""12-hour-ahead significant wave height forecasting.

Top-level exports cover the common workflow: load data, build target,
split, engineer features, fit baselines or regressors, score.

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
    SeasonalNaiveForecaster,
)
from .config import (
    CIRCULAR_COLS,
    FEATURE_COLS,
    HORIZON_HOURS,
    HORIZON_STEPS,
    SAMPLING_FREQ_MINUTES,
    TARGET_COL,
)
from .data import (
    NEIGHBOUR_FILES,
    WIND_FILES,
    chronological_split,
    load_data,
    load_neighbours,
    load_wind,
    make_target,
    restrict_to_overlap,
)
from .evaluate import EvaluationResult, compare, evaluate, mean_impute, scale_features
from .experiments import (
    compose_run_name,
    evaluate_and_log,
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
    # data
    "NEIGHBOUR_FILES",
    "WIND_FILES",
    "chronological_split",
    "load_data",
    "load_neighbours",
    "load_wind",
    "make_target",
    "restrict_to_overlap",
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
    "SeasonalNaiveForecaster",
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
    "mean_impute",
    "scale_features",
    # experiments
    "compose_run_name",
    "evaluate_and_log",
    "log_run",
    "read_log",
    "recent_runs",
    "wind_tag",
]
