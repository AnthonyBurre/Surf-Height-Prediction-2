"""12-hour-ahead significant wave height forecasting.

Top-level exports cover the common workflow: load data, build target,
split, engineer features, fit baselines or regressors, score.

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
from .features import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
    encode_circular,
)
from .metrics import bias, mae, rmse, skill_score, summarise

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
    "add_lag_features",
    "add_rolling_features",
    "add_time_features",
    "encode_circular",
    # baselines
    "ClimatologyHourForecaster",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
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
]
