"""12-hour-ahead significant wave height forecasting.

Top-level exports cover the common workflow: load data, build target,
split, engineer features, fit baselines or regressors, score.

Sequence-model forecasters (``SimpleRNNForecaster``, ``GRUForecaster``,
``LSTMForecaster``, ``TCNForecaster``) require ``torch`` and are loaded
lazily — ``import forecast`` stays light, and the first reference to one
of those names triggers the torch import (with a clear ImportError if
torch is not installed).

Example
-------
>>> from forecast import load_data, make_target, chronological_split
>>> from forecast import PersistenceForecaster, evaluate
>>> df = load_data()
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
from .data import chronological_split, load_data, make_target
from .evaluate import EvaluationResult, compare, evaluate
from .experiments import evaluate_and_log, log_run, read_log
from .features import (
    FeatureConfig,
    add_lag_features,
    add_momentum,
    add_neighbour_features,
    add_rolling_features,
    add_time_features,
    build_mooloolaba_features,
    build_seq_features,
    encode_circular,
)
from .metrics import bias, mae, rmse, skill_score, summarise

_NEURAL_NAMES = frozenset({
    "GRUForecaster",
    "LSTMForecaster",
    "SimpleRNNForecaster",
    "TCNForecaster",
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
    "chronological_split",
    "load_data",
    "make_target",
    # features
    "FeatureConfig",
    "add_lag_features",
    "add_momentum",
    "add_neighbour_features",
    "add_rolling_features",
    "add_time_features",
    "build_mooloolaba_features",
    "build_seq_features",
    "encode_circular",
    # baselines
    "ClimatologyHourForecaster",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    # neural
    "GRUForecaster",
    "LSTMForecaster",
    "SimpleRNNForecaster",
    "TCNForecaster",
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
    # experiments
    "evaluate_and_log",
    "log_run",
    "read_log",
]
