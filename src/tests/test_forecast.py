import numpy as np
import pandas as pd
import pytest

from forecast import (
    ClimatologyHourForecaster,
    HORIZON_STEPS,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    bias,
    chronological_split,
    compare,
    encode_circular,
    evaluate,
    mae,
    make_target,
    rmse,
    skill_score,
    summarise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_df(n: int = 200, freq: str = "30min", seed: int = 0) -> pd.DataFrame:
    """Build a regular 30-min grid of fake wave-buoy-like data.

    A deterministic shape + small noise is enough to exercise lag/rolling
    features and give models something learnable.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    t = np.arange(n)
    diurnal = 0.5 * np.sin(2 * np.pi * t / 48)  # 24h cycle
    df = pd.DataFrame(
        {
            "hsig_m": 1.2 + diurnal + 0.05 * rng.standard_normal(n),
            "hmax_m": 2.0 + diurnal + 0.1 * rng.standard_normal(n),
            "tz_s": 5.5 + 0.2 * rng.standard_normal(n),
            "tp_s": 9.0 + 0.3 * rng.standard_normal(n),
            "peak_dir_deg": (90 + 5 * rng.standard_normal(n)) % 360,
            "sst_c": 25.0 + 0.1 * rng.standard_normal(n),
        },
        index=idx,
    )
    df.index.name = "datetime_utc"
    return df


# ---------------------------------------------------------------------------
# data.make_target
# ---------------------------------------------------------------------------


def test_make_target_shifts_by_horizon():
    df = _synthetic_df(100)
    y = make_target(df, horizon_steps=HORIZON_STEPS)
    # y at t should equal hsig_m at t + HORIZON_STEPS
    assert y.iloc[0] == pytest.approx(df["hsig_m"].iloc[HORIZON_STEPS])
    assert y.iloc[50] == pytest.approx(df["hsig_m"].iloc[50 + HORIZON_STEPS])


def test_make_target_tail_is_nan():
    df = _synthetic_df(100)
    y = make_target(df, horizon_steps=HORIZON_STEPS)
    # The last HORIZON_STEPS values have no future observation to target.
    assert y.iloc[-HORIZON_STEPS:].isna().all()
    assert not y.iloc[:-HORIZON_STEPS].isna().any()


def test_make_target_name_reflects_horizon():
    df = _synthetic_df(50)
    y = make_target(df, horizon_steps=6, target_col="hsig_m")
    assert y.name == "hsig_m_plus_6"


def test_make_target_custom_column():
    df = _synthetic_df(50)
    y = make_target(df, horizon_steps=2, target_col="sst_c")
    assert y.iloc[0] == pytest.approx(df["sst_c"].iloc[2])


# ---------------------------------------------------------------------------
# data.chronological_split
# ---------------------------------------------------------------------------


def test_chronological_split_sizes():
    df = _synthetic_df(100)
    y = make_target(df, horizon_steps=2)
    Xtr, Xte, ytr, yte = chronological_split(df, y, test_frac=0.2)
    assert len(Xtr) == 80
    assert len(Xte) == 20
    assert len(ytr) == 80
    assert len(yte) == 20


def test_chronological_split_preserves_order():
    df = _synthetic_df(100)
    y = make_target(df, horizon_steps=2)
    Xtr, Xte, _, _ = chronological_split(df, y, test_frac=0.2)
    # All training timestamps must precede all test timestamps.
    assert Xtr.index.max() < Xte.index.min()


def test_chronological_split_x_and_y_align():
    df = _synthetic_df(100)
    y = make_target(df, horizon_steps=2)
    Xtr, Xte, ytr, yte = chronological_split(df, y, test_frac=0.3)
    assert (Xtr.index == ytr.index).all()
    assert (Xte.index == yte.index).all()


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_chronological_split_rejects_invalid_frac(bad):
    df = _synthetic_df(50)
    y = make_target(df, horizon_steps=2)
    with pytest.raises(ValueError):
        chronological_split(df, y, test_frac=bad)


# ---------------------------------------------------------------------------
# features — leakage + correctness
# ---------------------------------------------------------------------------


def test_add_lag_features_values_match_shift():
    df = _synthetic_df(20)
    out = add_lag_features(df, columns=["hsig_m"], lags=[1, 3])
    # lag_k at row i should equal hsig_m at row i-k
    assert out["hsig_m_lag_1"].iloc[1] == pytest.approx(df["hsig_m"].iloc[0])
    assert out["hsig_m_lag_3"].iloc[5] == pytest.approx(df["hsig_m"].iloc[2])
    # first k rows are NaN
    assert out["hsig_m_lag_3"].iloc[:3].isna().all()


def test_add_rolling_features_are_shifted_by_one():
    """Rolling features must not include the current observation — that would
    leak the label at prediction time."""
    df = _synthetic_df(30)
    out = add_rolling_features(df, columns=["hsig_m"], windows=[4], stats=("mean",))
    # shift(1) then rolling(4) at row 10 aggregates the shifted series'
    # positions 7..10, which are the unshifted series' positions 6..9.
    expected = df["hsig_m"].iloc[6:10].mean()
    assert out["hsig_m_roll4_mean"].iloc[10] == pytest.approx(expected)


def test_add_rolling_features_rejects_unknown_stat():
    df = _synthetic_df(10)
    with pytest.raises(ValueError, match="Unknown stats"):
        add_rolling_features(df, columns=["hsig_m"], windows=[4], stats=("zscore",))


def test_add_time_features_cyclical_bounds():
    # 49 rows of 30-min data covers 24h plus the next 00:00 timestamp.
    df = _synthetic_df(49)
    out = add_time_features(df)
    assert ((out["hour_sin"] >= -1.0) & (out["hour_sin"] <= 1.0)).all()
    assert ((out["hour_cos"] >= -1.0) & (out["hour_cos"] <= 1.0)).all()
    # Over a full day the sin/cos trace should loop back to the starting value.
    assert out["hour_sin"].iloc[0] == pytest.approx(out["hour_sin"].iloc[48], abs=1e-9)
    assert out["hour_cos"].iloc[0] == pytest.approx(out["hour_cos"].iloc[48], abs=1e-9)


def test_encode_circular_replaces_original_column():
    df = _synthetic_df(20)
    out = encode_circular(df)
    assert "peak_dir_deg" not in out.columns
    assert "peak_dir_deg_sin" in out.columns
    assert "peak_dir_deg_cos" in out.columns
    # sin^2 + cos^2 == 1 for all rows
    ss = out["peak_dir_deg_sin"] ** 2 + out["peak_dir_deg_cos"] ** 2
    assert np.allclose(ss, 1.0)


def test_encode_circular_handles_360_equals_0():
    df = pd.DataFrame({"peak_dir_deg": [0.0, 360.0]}, index=pd.date_range("2020-01-01", periods=2, freq="30min"))
    out = encode_circular(df)
    assert out["peak_dir_deg_sin"].iloc[0] == pytest.approx(out["peak_dir_deg_sin"].iloc[1])
    assert out["peak_dir_deg_cos"].iloc[0] == pytest.approx(out["peak_dir_deg_cos"].iloc[1])


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def test_persistence_predict_equals_current_target():
    df = _synthetic_df(50)
    y = make_target(df, horizon_steps=2)
    model = PersistenceForecaster().fit(df, y)
    preds = model.predict(df)
    np.testing.assert_allclose(preds, df["hsig_m"].to_numpy())


def test_persistence_perfect_on_constant_series():
    """A flat series is exactly forecastable by persistence."""
    idx = pd.date_range("2020-01-01", periods=100, freq="30min", tz="UTC")
    df = pd.DataFrame({"hsig_m": 1.5}, index=idx)
    y = make_target(df, horizon_steps=24)
    preds = PersistenceForecaster().fit(df, y).predict(df)
    mask = ~y.isna()
    np.testing.assert_allclose(preds[mask], y[mask])


def test_seasonal_naive_looks_up_same_hour_prior_period():
    df = _synthetic_df(200)
    y = make_target(df, horizon_steps=24)
    model = SeasonalNaiveForecaster(period_steps=48, horizon_steps=24).fit(df, y)
    preds = model.predict(df)
    # lookback = 48 - 24 = 24 steps
    assert preds[50] == pytest.approx(df["hsig_m"].iloc[50 - 24])
    # first lookback rows are NaN
    assert np.isnan(preds[:24]).all()


def test_seasonal_naive_rejects_period_le_horizon():
    with pytest.raises(ValueError, match="period_steps"):
        SeasonalNaiveForecaster(period_steps=24, horizon_steps=24)


def test_climatology_hour_predicts_training_hourly_mean():
    # Build a series whose hsig_m depends only on hour-of-day.
    n = 48 * 10  # 10 days of 30-min data
    idx = pd.date_range("2020-01-01", periods=n, freq="30min", tz="UTC")
    hsig = idx.hour.to_numpy(dtype=float)  # equals the hour-of-day
    df = pd.DataFrame({"hsig_m": hsig}, index=idx)
    y = make_target(df, horizon_steps=24)  # 12h ahead
    model = ClimatologyHourForecaster(horizon_steps=24).fit(df, y)
    preds = model.predict(df.iloc[:48])
    # For the first row (t=00:00), forecast time is 12:00, so prediction = 12.
    assert preds[0] == pytest.approx(12.0)
    # For row at t=06:00 (index 12), forecast time is 18:00, so prediction = 18.
    assert preds[12] == pytest.approx(18.0)


def test_climatology_predict_before_fit_raises():
    df = _synthetic_df(50)
    with pytest.raises(RuntimeError):
        ClimatologyHourForecaster().predict(df)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_mae_rmse_bias_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.0])
    assert mae(y_true, y_pred) == pytest.approx((0.5 + 0 + 1.0) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((0.25 + 0 + 1.0) / 3))
    assert bias(y_true, y_pred) == pytest.approx((0.5 + 0 - 1.0) / 3)


def test_metrics_ignore_nan():
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, np.nan, 5.0])
    # Only rows 0 and 3 are finite in both.
    assert mae(y_true, y_pred) == pytest.approx((0 + 1.0) / 2)


def test_skill_score_zero_when_model_matches_baseline():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert skill_score(y_true, y_pred, y_pred) == pytest.approx(0.0)


def test_skill_score_one_when_model_is_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    baseline = np.array([0.0, 0.0, 0.0])
    assert skill_score(y_true, y_true, baseline) == pytest.approx(1.0)


def test_skill_score_negative_when_model_worse_than_baseline():
    y_true = np.array([1.0, 2.0, 3.0])
    baseline = np.array([1.1, 2.0, 2.9])  # small errors
    model_preds = np.array([5.0, 5.0, 5.0])  # big errors
    assert skill_score(y_true, model_preds, baseline) < 0


def test_summarise_includes_skill_only_when_baseline_given():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    out = summarise(y_true, y_pred)
    assert "SkillVsBaseline" not in out
    out2 = summarise(y_true, y_pred, y_pred_baseline=np.array([0.0, 0.0, 0.0]))
    assert "SkillVsBaseline" in out2


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_metrics_and_predictions():
    df = _synthetic_df(200)
    y = make_target(df, horizon_steps=4)
    Xtr, Xte, ytr, yte = chronological_split(df, y, test_frac=0.25)
    result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="persistence")
    assert result.name == "persistence"
    assert {"MAE", "RMSE", "Bias"} <= set(result.metrics)
    assert len(result.predictions) == len(Xte)


def test_evaluate_masks_nan_training_rows():
    """A lag feature with NaNs in the first rows shouldn't crash model.fit."""
    df = _synthetic_df(200)
    X = add_lag_features(df, columns=["hsig_m"], lags=[10])
    y = make_target(df, horizon_steps=4)
    Xtr, Xte, ytr, yte = chronological_split(X, y, test_frac=0.25)
    from sklearn.linear_model import LinearRegression
    result = evaluate(LinearRegression(), Xtr, ytr, Xte, yte, name="lr")
    assert not np.isnan(result.metrics["MAE"])


def test_compare_sorts_by_rmse():
    df = _synthetic_df(200)
    y = make_target(df, horizon_steps=4)
    Xtr, Xte, ytr, yte = chronological_split(df, y, test_frac=0.25)
    r1 = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="a")
    r2 = evaluate(ClimatologyHourForecaster(horizon_steps=4), Xtr, ytr, Xte, yte, name="b")
    table = compare([r1, r2])
    assert list(table.index) == sorted(table.index, key=lambda m: table.loc[m, "RMSE"])
