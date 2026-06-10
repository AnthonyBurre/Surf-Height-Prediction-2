import numpy as np
import pandas as pd

from forecast.constants import HORIZON_STEPS
from forecast.targets import align_xy, make_target, make_targets


def test_make_target_reads_h_steps_ahead(synthetic_series):
    y = synthetic_series
    h = 24
    tgt = make_target(y, h)
    steps = HORIZON_STEPS[h]
    # row at origin t holds y(t + steps)
    for t in (200, 7000, 15000):
        assert np.isclose(tgt.iloc[t], y.iloc[t + steps], equal_nan=True)


def test_make_target_tail_is_nan(synthetic_series):
    y = synthetic_series
    steps = HORIZON_STEPS[72]
    tgt = make_target(y, 72)
    assert tgt.iloc[-steps:].isna().all()


def test_make_targets_columns(synthetic_series):
    cols = make_targets(synthetic_series).columns.tolist()
    assert cols == [f"y_h{h}" for h in (6, 12, 24, 36, 48, 72)]


def test_align_xy_drops_missing_label_and_required(synthetic_series):
    y = synthetic_series
    tgt = make_target(y, 12)
    X = pd.DataFrame({"hsig_m": y, "feat": np.arange(len(y), dtype=float)}, index=y.index)
    Xa, ya = align_xy(X, tgt)
    assert not ya.isna().any()
    assert not Xa["hsig_m"].isna().any()   # required col enforced
    # every surviving origin has a finite future label
    assert (Xa.index == ya.index).all()


def test_align_xy_keeps_nonrequired_nans_for_imputation(synthetic_series):
    y = synthetic_series
    tgt = make_target(y, 12)
    X = pd.DataFrame({"hsig_m": y, "lag": y.shift(144)}, index=y.index)  # long lag has NaNs early
    Xa, ya = align_xy(X, tgt)
    assert Xa["lag"].isna().any()  # non-required NaNs survive (model imputes)
