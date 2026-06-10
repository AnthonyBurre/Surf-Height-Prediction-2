"""Forecasting harness for Mooloolaba significant wave height.

Import as ``import forecast as fc``. The public surface:

- data/targets/features:  ``build_dataset``, ``load_target``, ``make_targets``,
  ``align_xy``, ``build_feature_matrix``
- baselines & models:     ``Persistence``, ``SeasonalNaive``, …, ``RidgeForecaster``,
  ``HGBForecaster``, ``DirectMultiHorizon``
- evaluation:             ``evaluate``, ``evaluate_and_log``, ``log_run``, ``read_log``,
  ``EvalResult``, and the ``fc.backtest`` submodule
  (``rolling_origin`` / ``block_bootstrap_ci`` / ``paired_block_bootstrap``)
- splitting:              ``RollingOriginSplitter``, ``blind_split``, ``FixedSplit``

``fc.neural`` (torch sequence models) is imported lazily so the package still
imports without the ``forecast`` extra.
"""
from . import backtest
from .baselines import (
    DriftRandomWalk, Persistence, SeasonalMean, SeasonalNaive, Theta, all_baselines,
)
from .constants import (
    BLIND_START, HORIZON_STEPS, HORIZONS_H, NEIGHBOUR_BUOYS, TARGET_BUOY, TARGET_COL,
    WIND_PAIR,
)
from .data import (
    available_sources, build_dataset, common_overlap, load_target, load_wave, load_wind,
)
from .evaluate import (
    EvalResult, evaluate, evaluate_and_log, log_run, read_log,
)
from .features import build_feature_matrix, target_value_cols
from .metrics import bias, mae, mase, rmse, skill, smape, wape
from .models import (
    DirectMultiHorizon, ElasticNetForecaster, HGBForecaster, LassoForecaster,
    RidgeForecaster, SklearnForecaster,
)
from .splits import (
    FixedSplit, RollingOriginSplitter, blind_split, chronological_split,
)
from .targets import align_xy, make_target, make_targets, residual_target


def __getattr__(name):
    # Lazy: `fc.neural` triggers the torch import only on access. Use
    # importlib (not `from . import neural`) so we don't re-enter __getattr__.
    if name == "neural":
        import importlib
        return importlib.import_module("forecast.neural")
    raise AttributeError(f"module 'forecast' has no attribute {name!r}")


__all__ = [
    "backtest", "neural",
    "build_dataset", "load_wave", "load_wind", "load_target", "available_sources",
    "common_overlap",
    "make_target", "make_targets", "residual_target", "align_xy",
    "build_feature_matrix", "target_value_cols",
    "Persistence", "SeasonalNaive", "SeasonalMean", "DriftRandomWalk", "Theta",
    "all_baselines",
    "RidgeForecaster", "LassoForecaster", "ElasticNetForecaster", "HGBForecaster",
    "SklearnForecaster", "DirectMultiHorizon",
    "rmse", "mae", "bias", "mase", "smape", "wape", "skill",
    "RollingOriginSplitter", "blind_split", "chronological_split", "FixedSplit",
    "evaluate", "evaluate_and_log", "log_run", "read_log", "EvalResult",
    "HORIZONS_H", "HORIZON_STEPS", "TARGET_COL", "TARGET_BUOY", "BLIND_START",
    "NEIGHBOUR_BUOYS", "WIND_PAIR",
]
