"""Origin-indexed target construction.

Every target row is keyed by the forecast **origin** `t` (not the target time
`t+h`). ``make_target(y, h)[t]`` holds ``y(t + steps)`` where
``steps = HORIZON_STEPS[h]``. Indexing at the origin matches production use
(you stand at `t` and predict the future) and makes leakage trivial to reason
about: features at row `t` may use data up to `t`, the label looks ahead.
"""
from typing import Sequence

import pandas as pd

from .constants import HORIZON_STEPS, HORIZONS_H, TARGET_COL


def make_target(y: pd.Series, horizon_h: int) -> pd.Series:
    """``y`` shifted so row `t` carries the value at `t + h`.

    The last ``steps`` origins have no observable future and become NaN; callers
    drop them at alignment (:func:`align_xy`).
    """
    steps = HORIZON_STEPS[horizon_h]
    return y.shift(-steps).rename(f"y_h{horizon_h}")


def make_targets(y: pd.Series, horizons: Sequence[int] = HORIZONS_H) -> pd.DataFrame:
    """One origin-indexed target column per horizon (``y_h6``, ``y_h12`` …)."""
    return pd.concat([make_target(y, h) for h in horizons], axis=1)


def residual_target(y: pd.Series, baseline_pred: pd.Series, horizon_h: int) -> pd.Series:
    """Residual-on-baseline target: ``y(t+h) - baseline(t)``.

    ``baseline_pred`` must already be origin-indexed (e.g. a persistence or
    seasonal-naive forecast for the same horizon). Letting the baseline carry
    the level and the model learn only the delta makes skill native.
    """
    tgt = make_target(y, horizon_h)
    return (tgt - baseline_pred.reindex(tgt.index)).rename(f"resid_h{horizon_h}")


def align_xy(
    X: pd.DataFrame,
    y: pd.Series,
    required_cols: Sequence[str] | None = (TARGET_COL,),
) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-join X and y on origin; drop rows missing the label or a required input.

    The single choke point guaranteeing features and labels share origins. By
    default a row survives if the target *and* the contemporaneous target value
    (``required_cols``) are present — these are exactly the origins where the
    persistence baseline can also predict, so models and baselines compare on the
    same rows. Remaining NaNs in other features are left for the model pipeline
    to impute (linear) or handle natively (HGB). Pass ``required_cols=None`` to
    drop only label-missing rows.
    """
    joined = X.join(y.rename("__y__"), how="inner")
    subset = ["__y__"]
    if required_cols:
        subset += [c for c in required_cols if c in joined.columns]
    joined = joined.dropna(subset=subset)
    y_out = joined.pop("__y__").rename(y.name)
    return joined, y_out
