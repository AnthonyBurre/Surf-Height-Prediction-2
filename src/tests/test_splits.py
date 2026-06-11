import pandas as pd

from forecast.constants import SOURCE_TZ
from forecast.splits import (
    FixedSplit, RollingOriginSplitter, blind_split, chronological_split,
)


def _idx(n=20000):
    return pd.date_range("2018-01-01", periods=n, freq="30min", tz=SOURCE_TZ)


def test_chronological_split_is_ordered_with_embargo():
    idx = _idx()
    s = chronological_split(idx, 0.7, 0.15, embargo_steps=48)
    assert s.train.max() < s.val.min() < s.test.min()
    # embargo gap between train and val
    assert (s.val.min() - s.train.max()) >= pd.Timedelta("30min") * 48


def test_blind_split_reserves_2025():
    idx = pd.date_range("2024-06-01", "2025-03-01", freq="30min", tz=SOURCE_TZ)
    dev, blind = blind_split(idx, blind_start="2025-01-01", embargo_steps=48)
    assert dev.max() < pd.Timestamp("2025-01-01", tz=SOURCE_TZ)
    assert blind.min() >= pd.Timestamp("2025-01-01", tz=SOURCE_TZ)
    assert (blind.min() - dev.max()) >= pd.Timedelta("30min") * 48


def test_rolling_origin_expanding_grows_and_no_overlap():
    idx = _idx()
    spl = RollingOriginSplitter(n_folds=4, val_size=1000, window="expanding", embargo_steps=48)
    prev_train_len = -1
    for train, val in spl.split(idx):
        assert train.max() < val.min()                 # train strictly before val
        assert (val.min() - train.max()) >= pd.Timedelta("30min") * 48  # embargo
        assert len(train) > prev_train_len             # expanding
        prev_train_len = len(train)


def test_rolling_origin_raises_when_too_many_folds():
    idx = _idx(1000)
    spl = RollingOriginSplitter(n_folds=5, val_size=300)
    try:
        list(spl.split(idx))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_fixed_split_yields_bound_indices():
    idx = _idx(1000)
    spl = FixedSplit(idx[:600], idx[600:])
    folds = list(spl.split())
    assert len(folds) == 1
    assert folds[0][0].equals(idx[:600]) and folds[0][1].equals(idx[600:])
