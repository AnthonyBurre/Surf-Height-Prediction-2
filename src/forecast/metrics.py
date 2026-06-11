"""Point-forecast KPIs (Phase 10).

All metrics take aligned, origin-indexed ``y_true``/``y_pred`` (Series or
arrays) and are NaN-safe. Skill and MASE are the portable headlines: the
upstream data is periodically revised, so absolute RMSE is *not* comparable
across data vintages while skill-vs-baseline largely is.
"""
import numpy as np
import pandas as pd

ArrayLike = "pd.Series | np.ndarray"


def _to_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


def rmse(y_true, y_pred) -> float:
    yt, yp = _to_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")


def mae(y_true, y_pred) -> float:
    yt, yp = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean signed error ``pred - true`` (positive = over-prediction)."""
    yt, yp = _to_arrays(y_true, y_pred)
    return float(np.mean(yp - yt)) if yt.size else float("nan")


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent; guards the zero-denominator blow-up."""
    yt, yp = _to_arrays(y_true, y_pred)
    if not yt.size:
        return float("nan")
    denom = np.abs(yt) + np.abs(yp)
    nz = denom > 0
    return float(np.mean(2.0 * np.abs(yp - yt)[nz] / denom[nz]) * 100.0)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|err| / sum|true|."""
    yt, yp = _to_arrays(y_true, y_pred)
    tot = np.abs(yt).sum()
    return float(np.abs(yp - yt).sum() / tot) if tot > 0 else float("nan")


def mase(y_true, y_pred, y_insample: ArrayLike, m: int = 1) -> float:
    """MAE scaled by the in-sample ``m``-step naive MAE (M-competition standard).

    ``y_insample`` is the training target series used to compute the naive
    scale; ``m=1`` is the one-step naive (random walk).
    """
    yt, yp = _to_arrays(y_true, y_pred)
    ins = np.asarray(y_insample, dtype=float)
    ins = ins[~np.isnan(ins)]
    if ins.size <= m or not yt.size:
        return float("nan")
    scale = np.mean(np.abs(ins[m:] - ins[:-m]))
    return float(np.mean(np.abs(yt - yp)) / scale) if scale > 0 else float("nan")


def skill(err_model: float, err_baseline: float) -> float:
    """``1 - err_model / err_baseline``; positive means the model beats the baseline."""
    if err_baseline is None or err_baseline == 0 or np.isnan(err_baseline):
        return float("nan")
    return float(1.0 - err_model / err_baseline)


def skill_from_preds(y_true, y_pred, y_pred_baseline, metric=rmse) -> float:
    """Skill computed directly from predictions, scored on the same rows."""
    return skill(metric(y_true, y_pred), metric(y_true, y_pred_baseline))


def stratified(y_true: pd.Series, y_pred: pd.Series, by: pd.Series, metric=rmse) -> pd.Series:
    """Compute ``metric`` within each group of ``by`` (e.g. calm/storm, season)."""
    df = pd.DataFrame({"yt": y_true, "yp": y_pred, "g": by}).dropna(subset=["yt", "yp"])
    return df.groupby("g", observed=True).apply(
        lambda d: metric(d["yt"], d["yp"]), include_groups=False
    )


METRIC_REGISTRY = {
    "rmse": rmse, "mae": mae, "bias": bias, "smape": smape, "wape": wape,
}
DEFAULT_METRICS = ("rmse", "mae", "bias")
