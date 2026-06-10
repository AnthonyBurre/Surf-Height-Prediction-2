"""Raw windowed channels — the second representation, for sequence models.

Where the engineered matrix suits linear/tree models, sequence nets consume raw
``(context_len, n_channels)`` windows ending at the origin `t`, paired with the
target ``y(t + steps)``. Short gaps are forward-filled; windows still containing
a NaN (in context or target) are dropped.
"""
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import HORIZON_STEPS


def make_windows(
    df: pd.DataFrame,
    cols: Sequence[str],
    context_len: int,
    horizon_h: int | Sequence[int],
    *,
    target_col: str,
    ffill_limit: int = 6,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build ``(X[n, context_len, n_channels], y, origin_index)``.

    ``horizon_h`` may be a single horizon or a sequence (joint multi-output:
    ``y`` is ``(n, n_horizons)``). Windows end at the origin `t`; ``y`` reads the
    target ``steps`` ahead. Rows with any remaining NaN after a short forward-fill
    are dropped so the tensors are clean.
    """
    cols = list(cols)
    horizons = [horizon_h] if isinstance(horizon_h, int) else list(horizon_h)
    feat = df[cols].ffill(limit=ffill_limit)
    tgt = df[target_col]

    values = feat.to_numpy(dtype=np.float32)
    target_arr = tgt.to_numpy(dtype=np.float32)
    n = len(df)
    max_steps = max(HORIZON_STEPS[h] for h in horizons)

    origins = range(context_len - 1, n - max_steps, stride)
    X, Y, idx = [], [], []
    for t in origins:
        win = values[t - context_len + 1 : t + 1]
        ys = [target_arr[t + HORIZON_STEPS[h]] for h in horizons]
        if np.isnan(win).any() or np.isnan(ys).any():
            continue
        X.append(win)
        Y.append(ys)
        idx.append(t)

    if not X:
        return (np.empty((0, context_len, len(cols)), np.float32),
                np.empty((0, len(horizons)), np.float32),
                df.index[:0])
    Xa = np.stack(X)
    Ya = np.asarray(Y, dtype=np.float32)
    if isinstance(horizon_h, int):
        Ya = Ya[:, 0]
    return Xa, Ya, df.index[list(idx)]


def windows_for_index(
    df: pd.DataFrame,
    cols: Sequence[str],
    context_len: int,
    origins: pd.DatetimeIndex,
    *,
    ffill_limit: int = 6,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Build context windows ending at each origin in ``origins``.

    Returns ``(X[n, context_len, n_channels], kept_origins)``; origins whose
    window still has a NaN after a short forward-fill (or that lack enough
    history) are dropped. Used by the sequence-NN adapter so it can build windows
    for exactly the fold origins the harness hands it.
    """
    cols = list(cols)
    values = df[cols].ffill(limit=ffill_limit).to_numpy(dtype=np.float32)
    pos = df.index.get_indexer(origins)
    X, kept = [], []
    for origin, t in zip(origins, pos):
        if t < context_len - 1:
            continue
        win = values[t - context_len + 1 : t + 1]
        if np.isnan(win).any():
            continue
        X.append(win)
        kept.append(origin)
    if not X:
        return np.empty((0, context_len, len(cols)), np.float32), origins[:0]
    return np.stack(X), pd.DatetimeIndex(kept)


class WindowScaler:
    """Per-channel robust scaler (median / IQR), fit on training windows only."""

    def __init__(self):
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "WindowScaler":
        flat = X.reshape(-1, X.shape[-1])
        self.center_ = np.nanmedian(flat, axis=0)
        q1, q3 = np.nanpercentile(flat, [25, 75], axis=0)
        iqr = q3 - q1
        self.scale_ = np.where(iqr > 0, iqr, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
