"""Append-only JSONL experiment log.

One ``log_run`` call appends one self-describing record:

    {timestamp, git_sha, name, model_class, hyperparams,
     data_sources, n_features, train, test, metrics, extra}

JSONL (not CSV) because hyperparameters are nested and heterogeneous across
model families — flattening to columns produces ragged NaNs that obscure
which parameter belonged to which run. ``read_log`` round-trips back to a
DataFrame for filtering/sorting in pandas.

The default log path resolves to ``<repo>/experiments.jsonl`` so notebooks
and scripts share one file regardless of cwd. The file is meant to be
committed to git — it's small, append-only, and ``git_sha`` makes any row
reproducible by checkout.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import EvaluationResult, evaluate

DEFAULT_LOG = Path(__file__).parents[2] / "experiments.jsonl"


def _jsonable(v: Any) -> Any:
    """Coerce numpy / pandas / Path types to JSON-native equivalents."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _git_sha() -> str | None:
    """HEAD sha plus a ``-dirty`` suffix if the working tree has uncommitted
    changes. Returns None if not in a git repo or git isn't available."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD"],
        capture_output=True,
    ).returncode != 0
    return head + ("-dirty" if dirty else "")


def _model_hyperparams(model: Any) -> dict[str, Any]:
    """sklearn estimators expose ``get_params``; our forecasters keep
    constructor args as plain attributes. Either way, return a JSON-able
    dict of hyperparameter name → value."""
    if model is None:
        return {}
    if hasattr(model, "get_params"):
        params = model.get_params(deep=False)
    else:
        params = {k: v for k, v in vars(model).items() if not k.startswith("_")}
    return _jsonable(params)


def _index_window(idx: pd.Index) -> dict[str, Any]:
    if len(idx) == 0:
        return {"start": None, "end": None, "n": 0}
    return {
        "start": idx.min().isoformat(),
        "end":   idx.max().isoformat(),
        "n":     int(len(idx)),
    }


def log_run(
    result: EvaluationResult,
    *,
    data_sources: list[str],
    train_index: pd.Index,
    test_index: pd.Index,
    n_features: int | None = None,
    extra: dict | None = None,
    path: str | Path = DEFAULT_LOG,
) -> dict:
    """Append one record to ``experiments.jsonl``.

    Returns the logged record (also useful for assertions in tests).
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "name": result.name,
        "model_class": type(result.model).__name__ if result.model is not None else result.name,
        "hyperparams": _model_hyperparams(result.model),
        "data_sources": [str(s) for s in data_sources],
        "n_features": int(n_features) if n_features is not None else None,
        "train": _index_window(train_index),
        "test": _index_window(test_index),
        "metrics": _jsonable(result.metrics),
        "extra": _jsonable(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def evaluate_and_log(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    data_sources: list[str],
    name: str | None = None,
    baseline_preds: np.ndarray | None = None,
    extra: dict | None = None,
    path: str | Path = DEFAULT_LOG,
) -> EvaluationResult:
    """Convenience: ``evaluate(...)`` then ``log_run(...)`` in one call."""
    result = evaluate(
        model, X_train, y_train, X_test, y_test,
        name=name, baseline_preds=baseline_preds,
    )
    log_run(
        result,
        data_sources=data_sources,
        train_index=X_train.index,
        test_index=X_test.index,
        n_features=X_train.shape[1],
        extra=extra,
        path=path,
    )
    return result


def read_log(path: str | Path = DEFAULT_LOG) -> pd.DataFrame:
    """Load the JSONL log as a DataFrame. Empty if the file doesn't exist."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)
