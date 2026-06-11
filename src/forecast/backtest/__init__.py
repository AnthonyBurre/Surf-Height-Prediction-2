"""Rolling-origin backtesting and the bootstrap noise floor (Phases 4–5).

A single train/val split is one noisy draw. :func:`rolling_origin` walks several
consecutive held-out blocks and reports the mean across folds (selection signal)
plus the fold-to-fold spread (regime uncertainty). The bootstrap functions put a
confidence interval on the headline metric (:func:`block_bootstrap_ci`) and
resolve model-vs-model differences far better than either absolute metric by
pairing per-origin errors on identical rows (:func:`paired_block_bootstrap`).
Residuals are autocorrelated, so all bootstraps resample contiguous *blocks*.
"""
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..metrics import METRIC_REGISTRY

Forecaster = "object with .fit(X, y) -> self and .predict(X) -> Series/array"


# --------------------------------------------------------------------------- #
# Rolling-origin walk-forward
# --------------------------------------------------------------------------- #
@dataclass
class RollingOriginResult:
    per_fold: pd.DataFrame                       # rows = folds, cols = metrics
    mean: dict[str, float]
    fold_spread: dict[str, float]                # std across folds
    predictions: pd.DataFrame                    # origin-indexed: y_true, y_pred, fold
    meta: dict = field(default_factory=dict)

    @property
    def residuals(self) -> pd.Series:
        return (self.predictions["y_true"] - self.predictions["y_pred"]).rename("residual")


def _as_series(pred, index: pd.Index) -> pd.Series:
    if isinstance(pred, pd.Series):
        return pred.reindex(index)
    return pd.Series(np.asarray(pred, dtype=float), index=index)


def rolling_origin(
    model_factory: Callable[[int], "Forecaster"],
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
    *,
    metrics: Sequence[str] = ("rmse", "mae", "bias"),
    seeds: Sequence[int] = (0,),
    meta: dict | None = None,
) -> RollingOriginResult:
    """Walk-forward backtest of ``model_factory`` over ``splitter`` folds.

    ``model_factory(seed)`` returns a fresh forecaster; stochastic models are
    averaged over ``seeds`` (the seed spread can exceed a hyperparameter effect).
    ``X``/``y`` are aligned and origin-indexed for a single horizon.
    """
    metric_fns = {m: METRIC_REGISTRY[m] for m in metrics}
    rows, pred_frames = [], []

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(y.index)):
        # Folds come from the full origin universe; alignment may have dropped
        # rows, so intersect with what is actually available.
        train_idx = y.index.intersection(train_idx)
        val_idx = y.index.intersection(val_idx)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        Xtr, ytr = X.loc[train_idx], y.loc[train_idx]
        Xva, yva = X.loc[val_idx], y.loc[val_idx]
        seed_preds = []
        for seed in seeds:
            model = model_factory(seed)
            model.fit(Xtr, ytr)
            seed_preds.append(_as_series(model.predict(Xva), val_idx).to_numpy())
        with warnings.catch_warnings():  # gap origins are all-NaN -> NaN pred, expected
            warnings.simplefilter("ignore", RuntimeWarning)
            yhat = pd.Series(np.nanmean(seed_preds, axis=0), index=val_idx)
        rows.append({m: fn(yva, yhat) for m, fn in metric_fns.items()})
        pred_frames.append(pd.DataFrame({"y_true": yva, "y_pred": yhat, "fold": fold_i}))

    per_fold = pd.DataFrame(rows)
    predictions = pd.concat(pred_frames) if pred_frames else pd.DataFrame(
        columns=["y_true", "y_pred", "fold"]
    )
    return RollingOriginResult(
        per_fold=per_fold,
        mean={m: float(per_fold[m].mean()) for m in metrics},
        fold_spread={m: float(per_fold[m].std(ddof=0)) for m in metrics},
        predictions=predictions,
        meta={**(meta or {}), "n_folds": len(per_fold), "seeds": list(seeds)},
    )


# --------------------------------------------------------------------------- #
# Bootstrap noise floor
# --------------------------------------------------------------------------- #
@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    se: float
    alpha: float = 0.05


@dataclass
class PairedResult:
    delta: float          # mean loss(a) - loss(b);  < 0 => a better
    lo: float
    hi: float
    significant: bool
    alpha: float = 0.05


def _block_starts(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Start positions of contiguous blocks tiling an array of length ``n``."""
    n_blocks = int(np.ceil(n / block_len))
    return rng.integers(0, max(n - block_len + 1, 1), size=n_blocks)


def _resample(values: np.ndarray, starts: np.ndarray, block_len: int, n: int) -> np.ndarray:
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
    return values[idx]


def suggest_block_len(residuals, max_lag: int = 200, threshold: float = 0.1) -> int:
    """Block length from the residual autocorrelation decorrelation scale."""
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    r = r - r.mean()
    if r.size < 10:
        return 1
    var = np.dot(r, r)
    if var == 0:
        return 1
    for lag in range(1, min(max_lag, r.size - 1)):
        ac = np.dot(r[:-lag], r[lag:]) / var
        if abs(ac) < threshold:
            return max(2 * lag, 2)
    return min(max_lag, r.size // 2 or 1)


def _rmse_resid(r: np.ndarray) -> float:
    return float(np.sqrt(np.mean(r ** 2)))


def _mae_resid(r: np.ndarray) -> float:
    return float(np.mean(np.abs(r)))


RESIDUAL_REDUCERS = {"rmse": _rmse_resid, "mae": _mae_resid, "bias": lambda r: float(np.mean(-r))}


def block_bootstrap_ci(
    residuals,
    *,
    reducer="rmse",
    block_len: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Moving-block bootstrap CI for a metric of per-origin ``residuals``.

    ``residuals`` are ``y_true - y_pred``. ``reducer`` is a name in
    :data:`RESIDUAL_REDUCERS` (``"rmse"``/``"mae"``/``"bias"``) or a callable
    mapping a residual array to a scalar. Contiguous blocks longer than the
    autocorrelation scale keep the spread from being understated.
    """
    reduce = RESIDUAL_REDUCERS[reducer] if isinstance(reducer, str) else reducer
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), float("nan"), alpha)
    bl = min(block_len or suggest_block_len(r), n)

    point = reduce(r)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = reduce(_resample(r, _block_starts(n, bl, rng), bl, n))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(float(point), float(lo), float(hi), float(boots.std(ddof=1)), alpha)


def paired_block_bootstrap(
    residuals_a,
    residuals_b,
    *,
    loss: str = "squared",
    block_len: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    """Paired moving-block bootstrap of the per-origin loss difference (a − b).

    ``residuals_a``/``residuals_b`` must be aligned on identical origins (same
    rows). The same resampled block indices index both, so shared hard periods
    cancel and the *difference* is resolved tightly. ``significant`` iff the CI
    excludes zero. ``delta < 0`` means model *a* is better.
    """
    a = np.asarray(residuals_a, dtype=float)
    b = np.asarray(residuals_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = a.size
    if n == 0:
        return PairedResult(float("nan"), float("nan"), float("nan"), False, alpha)
    la = a ** 2 if loss == "squared" else np.abs(a)
    lb = b ** 2 if loss == "squared" else np.abs(b)
    d = la - lb
    bl = min(block_len or suggest_block_len(d), n)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = _block_starts(n, bl, rng)
        boots[i] = _resample(d, starts, bl, n).mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return PairedResult(float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0), alpha)


def diebold_mariano(residuals_a, residuals_b, *, loss: str = "squared", h: int = 1):
    """Diebold–Mariano test of equal predictive accuracy (a vs b).

    Returns ``(dm_stat, p_value)`` with a small-sample (Harvey) correction. A
    negative stat favours model *a*. Uses a two-sided normal approximation.
    """
    from scipy import stats

    a = np.asarray(residuals_a, dtype=float)
    b = np.asarray(residuals_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    d = (a ** 2 - b ** 2) if loss == "squared" else (np.abs(a) - np.abs(b))
    n = d.size
    if n < 8:
        return float("nan"), float("nan")
    dbar = d.mean()
    # long-run variance with (h-1) autocovariances
    gamma0 = np.dot(d - dbar, d - dbar) / n
    var = gamma0
    for k in range(1, h):
        cov = np.dot(d[:-k] - dbar, d[k:] - dbar) / n
        var += 2 * cov
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / n)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= harvey
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)
