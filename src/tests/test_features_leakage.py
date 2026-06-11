"""The load-bearing leakage guards: window features end at t-1 and no feature at
origin t depends on any value strictly after t."""
import numpy as np
import pandas as pd

from forecast.features import (
    build_feature_matrix, circular_encode, rolling_features,
)


def test_no_future_leakage(synthetic_frame):
    df = synthetic_frame
    X1 = build_feature_matrix(df, value_cols=["hsig_m", "hmax_m", "tz_s"])
    t0 = X1.index[8000]
    # perturb everything strictly after t0
    df2 = df.copy()
    future = df2.index > t0
    df2.loc[future, "hsig_m"] += 99.0
    df2.loc[future, "hmax_m"] += 99.0
    X2 = build_feature_matrix(df2, value_cols=["hsig_m", "hmax_m", "tz_s"])
    row1, row2 = X1.loc[t0], X2.loc[t0]
    assert np.allclose(row1.to_numpy(), row2.to_numpy(), equal_nan=True), \
        "a feature at origin t changed when only the future changed"


def test_rolling_ends_at_t_minus_1(synthetic_frame):
    df = synthetic_frame
    cols = ["hsig_m"]
    X1 = rolling_features(df, cols, windows=(6, 48), stats=("mean", "std"))
    t_pos = 8000
    t0 = df.index[t_pos]
    # perturb ONLY the current step t0
    df2 = df.copy()
    df2.iloc[t_pos, df2.columns.get_loc("hsig_m")] += 99.0
    X2 = rolling_features(df2, cols, windows=(6, 48), stats=("mean", "std"))
    # rolling stats at t0 exclude the current step -> unchanged
    assert np.allclose(X1.loc[t0].to_numpy(), X2.loc[t0].to_numpy(), equal_nan=True)
    # but the NEXT step's rolling stat (which now includes the perturbed t0) changes
    t1 = df.index[t_pos + 1]
    assert not np.allclose(X1.loc[t1].to_numpy(), X2.loc[t1].to_numpy(), equal_nan=True)


def test_current_value_feature_reflects_present(synthetic_frame):
    df = synthetic_frame
    X = build_feature_matrix(df, value_cols=["hsig_m"])
    t0 = df.index[8000]
    assert np.isclose(X.loc[t0, "hsig_m_now"], df.loc[t0, "hsig_m"])


def test_circular_encode_roundtrips_angles():
    deg = pd.Series([0.0, 90.0, 180.0, 270.0, 359.0])
    df = pd.DataFrame({"peak_dir_deg": deg})
    enc = circular_encode(df, ["peak_dir_deg"])
    rec = (np.rad2deg(np.arctan2(enc["peak_dir_deg_sin"], enc["peak_dir_deg_cos"])) % 360)
    assert np.allclose(rec.to_numpy(), deg.to_numpy(), atol=1e-6)
    assert "peak_dir_deg" not in enc.columns
