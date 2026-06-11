"""The evaluation harness + append-only experiment log (Phases 9–10, 13).

``evaluate`` ties the pieces together for one model across horizons: rolling-origin
backtest, a block-bootstrap CI on the headline metric, skill versus a baseline
scored on the **same rows**, and a paired bootstrap verdict against that baseline.
``evaluate_and_log`` is a drop-in that also appends to ``experiments.jsonl``;
``log_run`` logs results (or raw dicts) and ``read_log`` reads the file back as a
DataFrame in one line. The log is the results database — every chart reads it.
"""
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import baselines as _baselines
from .backtest import (
    block_bootstrap_ci, paired_block_bootstrap, rolling_origin, suggest_block_len,
)
from .constants import HORIZONS_H, LOG_PATH, TARGET_COL
from .metrics import rmse
from .targets import align_xy, make_target


def _versions() -> dict:
    import sklearn
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "sklearn": sklearn.__version__}
    try:
        import torch
        out["torch"] = torch.__version__
    except ImportError:
        pass
    return out


@dataclass
class EvalResult:
    """Canonical per-model result record (one model, many horizons)."""

    name: str
    horizons: list[int]
    metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    metric_ci: dict[int, dict[str, tuple[float, float]]] = field(default_factory=dict)
    fold_spread: dict[int, dict[str, float]] = field(default_factory=dict)
    skill: dict[int, dict[str, float]] = field(default_factory=dict)
    paired: dict[int, dict] = field(default_factory=dict)
    baseline_name: str = "persistence"
    predictions: dict[int, pd.DataFrame] = field(default_factory=dict, repr=False)
    data_sources: list[str] = field(default_factory=list)
    n_features: int = 0
    seeds: list[int] = field(default_factory=lambda: [0])
    n_folds: int = 0
    hyperparams: dict = field(default_factory=dict)
    mode: str = "select"
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_records(self) -> list[dict]:
        """One flat, JSON-serializable row per horizon (chart-friendly)."""
        rows = []
        for h in self.horizons:
            m = self.metrics.get(h, {})
            ci = self.metric_ci.get(h, {}).get("rmse", (float("nan"), float("nan")))
            pr = self.paired.get(h, {})
            rows.append({
                "name": self.name, "horizon_h": h, "mode": self.mode,
                "rmse": m.get("rmse"), "mae": m.get("mae"), "bias": m.get("bias"),
                "rmse_lo": ci[0], "rmse_hi": ci[1],
                "fold_spread_rmse": self.fold_spread.get(h, {}).get("rmse"),
                "skill_rmse": self.skill.get(h, {}).get("rmse"),
                "baseline": self.baseline_name,
                "paired_delta": pr.get("delta"), "paired_lo": pr.get("lo"),
                "paired_hi": pr.get("hi"), "paired_sig": pr.get("significant"),
                "n_folds": self.n_folds, "n_features": self.n_features,
                "data_sources": ",".join(self.data_sources), "seeds": str(self.seeds),
                "created": self.created, "versions": _versions(), "git_sha": None,
            })
        return rows


def evaluate(
    model_factory: Callable[[int, int], object],
    *,
    y_full: pd.Series,
    splitter,
    X: pd.DataFrame | None = None,
    horizons: Sequence[int] = HORIZONS_H,
    index: pd.DatetimeIndex | None = None,
    baseline_factory=_baselines.Persistence,
    baseline_name: str = "persistence",
    metrics: Sequence[str] = ("rmse", "mae", "bias"),
    seeds: Sequence[int] = (0,),
    name: str = "model",
    block_len: int | None = None,
    n_boot: int = 1000,
    data_sources: Sequence[str] = (),
    hyperparams: dict | None = None,
    paired_loss: str = "squared",
    mode: str = "select",
) -> EvalResult:
    """Backtest ``model_factory`` across ``horizons`` with skill + noise floor.

    ``model_factory(horizon_h, seed)`` returns a fresh forecaster. When ``X`` is
    given (engineered features) the model sees the aligned design matrix; when
    ``X`` is ``None`` the model only needs the origin index (baselines and the
    sequence NN, which bind their own source frame). ``index`` restricts the
    origin universe (e.g. the dev set). Skill and the paired test are computed on
    the rows the model and baseline share.
    """
    res = EvalResult(
        name=name, horizons=list(horizons), baseline_name=baseline_name,
        seeds=list(seeds), data_sources=list(data_sources),
        hyperparams=hyperparams or {}, mode=mode,
    )
    base_frame_full = pd.DataFrame({TARGET_COL: y_full})

    for h in horizons:
        yh = make_target(y_full, h)
        if X is not None:
            Xh, ya = align_xy(X, yh)
        else:
            Xh, ya = align_xy(base_frame_full, yh)
        if index is not None:
            keep = ya.index.isin(index)
            Xh, ya = Xh.loc[keep], ya.loc[keep]
        if len(ya) == 0:
            continue

        mr = rolling_origin(lambda s, hh=h: model_factory(hh, s), Xh, ya, splitter,
                            metrics=metrics, seeds=seeds)
        bframe = base_frame_full.reindex(ya.index)
        br = rolling_origin(lambda s, hh=h: baseline_factory(y_full, hh), bframe, ya,
                            splitter, metrics=("rmse",), seeds=(0,))

        res.metrics[h] = mr.mean
        res.fold_spread[h] = mr.fold_spread
        res.n_folds = mr.meta["n_folds"]
        res.n_features = Xh.shape[1] if X is not None else 0
        res.predictions[h] = mr.predictions

        # common rows for skill + paired test
        common = mr.predictions.index.intersection(br.predictions.index)
        rm = (mr.predictions["y_true"] - mr.predictions["y_pred"]).reindex(common)
        rb = (br.predictions["y_true"] - br.predictions["y_pred"]).reindex(common)
        bl = block_len or suggest_block_len(rm)
        ci = block_bootstrap_ci(rm, reducer="rmse", block_len=bl, n_boot=n_boot)
        res.metric_ci[h] = {"rmse": (ci.lo, ci.hi)}
        rmse_m = rmse(np.zeros_like(rm.dropna()), -rm.dropna())
        rmse_b = rmse(np.zeros_like(rb.dropna()), -rb.dropna())
        res.skill[h] = {"rmse": 1.0 - rmse_m / rmse_b if rmse_b else float("nan")}
        pr = paired_block_bootstrap(rm, rb, loss=paired_loss, block_len=bl, n_boot=n_boot)
        res.paired[h] = {"vs": baseline_name, "delta": pr.delta, "lo": pr.lo,
                         "hi": pr.hi, "significant": pr.significant}
    return res


# --------------------------------------------------------------------------- #
# Experiment log
# --------------------------------------------------------------------------- #
def log_run(result, *, log_path: Path = LOG_PATH, extra: dict | None = None) -> None:
    """Append one JSON line per horizon to ``experiments.jsonl``.

    ``result`` is an :class:`EvalResult`, a single record dict, or a list of
    record dicts (results computed outside the harness).
    """
    if isinstance(result, EvalResult):
        records = result.to_records()
    elif isinstance(result, dict):
        records = [result]
    else:
        records = list(result)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        for rec in records:
            if extra:
                rec = {**rec, **extra}
            f.write(json.dumps(rec, default=str) + "\n")


def evaluate_and_log(*args, log_path: Path = LOG_PATH, extra: dict | None = None,
                     **kwargs) -> EvalResult:
    """Drop-in for :func:`evaluate` that also appends the result to the log."""
    result = evaluate(*args, **kwargs)
    log_run(result, log_path=log_path, extra=extra)
    return result


def read_log(log_path: Path = LOG_PATH) -> pd.DataFrame:
    """Read ``experiments.jsonl`` back as a DataFrame (empty if no runs yet)."""
    log_path = Path(log_path)
    if not log_path.exists() or log_path.stat().st_size == 0:
        return pd.DataFrame(columns=["name", "horizon_h", "mode", "rmse", "skill_rmse"])
    return pd.read_json(log_path, lines=True)
