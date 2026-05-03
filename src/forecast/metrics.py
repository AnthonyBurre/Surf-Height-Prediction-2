"""Regression metrics that ignore NaN and work on numpy or pandas inputs."""
import numpy as np


def _align(*arrays) -> tuple[np.ndarray, ...]:
    """Coerce inputs to float arrays and mask out any row with a NaN.

    Accepts two or more arrays of identical shape; returns the same number
    of arrays, all clipped to the rows where every input was finite.
    """
    cast = [np.asarray(a, dtype=float) for a in arrays]
    shape = cast[0].shape
    for a in cast[1:]:
        if a.shape != shape:
            raise ValueError(f"shape mismatch: {[a.shape for a in cast]}")
    mask = np.ones(shape, dtype=bool)
    for a in cast:
        mask &= ~np.isnan(a)
    return tuple(a[mask] for a in cast)


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")


def rmse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean signed error; positive = systematic over-prediction."""
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(yp - yt)) if yt.size else float("nan")


def skill_score(y_true, y_pred, y_pred_baseline) -> float:
    """1 - MSE(model) / MSE(baseline). Positive ⇒ beats baseline, 1.0 = perfect.

    Using MSE (not MAE) because skill scores are conventionally defined on
    squared error, and it's what matches the RMSE that regression models
    optimise.
    """
    yt, yp, yb = _align(y_true, y_pred, y_pred_baseline)
    if not yt.size:
        return float("nan")
    mse_model = float(np.mean((yt - yp) ** 2))
    mse_base = float(np.mean((yt - yb) ** 2))
    if mse_base <= 1e-12:
        return float("nan")
    return 1.0 - mse_model / mse_base


def summarise(y_true, y_pred, y_pred_baseline=None) -> dict[str, float]:
    """One-shot metrics dict. Skill is included iff a baseline is provided."""
    out: dict[str, float] = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }
    if y_pred_baseline is not None:
        out["SkillVsBaseline"] = skill_score(y_true, y_pred, y_pred_baseline)
    return out
