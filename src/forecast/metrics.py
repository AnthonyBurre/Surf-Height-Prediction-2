"""Regression metrics that ignore NaN and work on numpy or pandas inputs."""
import numpy as np
import pandas as pd


def _align(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


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
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yb = np.asarray(y_pred_baseline, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp) | np.isnan(yb))
    if not mask.any():
        return float("nan")
    mse_model = float(np.mean((yt[mask] - yp[mask]) ** 2))
    mse_base = float(np.mean((yt[mask] - yb[mask]) ** 2))
    if mse_base == 0:
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
