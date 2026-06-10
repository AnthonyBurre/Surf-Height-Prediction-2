"""Chronological splitting and rolling-origin (walk-forward) folds (Phase 4).

Never shuffle: random K-fold leaks the future into the past through
autocorrelation. Folds you select on are **validation** (in scikit-learn's CV
vocabulary every held-out fold is confusingly called "test"). The genuinely
once-touched **test** is the pre-committed blind slice (:func:`blind_split`).

Every seam carries an **embargo** of ``embargo_steps`` (= the horizon in steps):
because the target looks ``h`` steps ahead, the last ``h`` training origins would
otherwise see into the held-out block.
"""
from dataclasses import dataclass
from typing import Iterator, Literal

import pandas as pd

from .constants import BLIND_START, SOURCE_TZ


@dataclass(frozen=True)
class Split:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex


def chronological_split(
    index: pd.DatetimeIndex,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    embargo_steps: int = 0,
) -> Split:
    """Ordered train/val/test split by position, with an embargo gap at each seam."""
    idx = index.sort_values()
    n = len(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = idx[: max(n_train - embargo_steps, 0)]
    val = idx[n_train : n_train + max(n_val - embargo_steps, 0)]
    test = idx[n_train + n_val :]
    return Split(train, val, test)


def blind_split(
    index: pd.DatetimeIndex,
    blind_start: str = BLIND_START,
    embargo_steps: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split into (dev, blind). Blind = everything from ``blind_start`` onward.

    The last ``embargo_steps`` dev origins are dropped so no dev target reaches
    into the blind period.
    """
    idx = index.sort_values()
    cutoff = pd.Timestamp(blind_start, tz=idx.tz or SOURCE_TZ)
    dev = idx[idx < cutoff]
    blind = idx[idx >= cutoff]
    if embargo_steps and len(dev):
        dev = dev[:-embargo_steps] if embargo_steps < len(dev) else dev[:0]
    return dev, blind


class RollingOriginSplitter:
    """Walk-forward folds with consecutive held-out blocks at the tail.

    ``n_folds`` validation blocks of ``val_size`` steps each occupy the tail of
    the index; fold *i*'s training window is everything before its block (minus
    the embargo). ``window="expanding"`` grows the train start from 0;
    ``"sliding"`` keeps a fixed ``train_size``.
    """

    def __init__(
        self,
        n_folds: int = 5,
        val_size: int = 5760,            # ~120 days of 30-min steps
        window: Literal["expanding", "sliding"] = "expanding",
        embargo_steps: int = 0,
        train_size: int | None = None,   # required for sliding
    ):
        self.n_folds = n_folds
        self.val_size = val_size
        self.window = window
        self.embargo_steps = embargo_steps
        self.train_size = train_size

    def split(self, index: pd.DatetimeIndex) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        idx = index.sort_values()
        n = len(idx)
        need = self.n_folds * self.val_size
        if need >= n:
            raise ValueError(
                f"n_folds*val_size={need} >= index length {n}; "
                f"reduce folds or val_size."
            )
        for i in range(self.n_folds):
            val_start = n - (self.n_folds - i) * self.val_size
            val_end = val_start + self.val_size
            train_end = val_start - self.embargo_steps
            if train_end <= 0:
                continue
            if self.window == "expanding":
                train_start = 0
            else:
                if self.train_size is None:
                    raise ValueError("sliding window requires train_size.")
                train_start = max(0, train_end - self.train_size)
            yield idx[train_start:train_end], idx[val_start:val_end]

    def n_splits(self) -> int:
        return self.n_folds


class FixedSplit:
    """A one-fold splitter yielding a pre-decided ``(train, val)`` pair.

    Used by the confirmatory pass: train on the whole dev set, score once on the
    blind slice. ``split`` ignores its argument and yields the bound indices.
    """

    def __init__(self, train: pd.DatetimeIndex, val: pd.DatetimeIndex):
        self._train = train
        self._val = val

    def split(self, index=None):
        yield self._train, self._val

    def n_splits(self) -> int:
        return 1
