import numpy as np
import pandas as pd

from forecast import metrics


def test_rmse_mae_bias_known_values():
    yt = pd.Series([1.0, 2.0, 3.0, 4.0])
    yp = pd.Series([1.0, 2.0, 4.0, 6.0])  # errors: 0,0,+1,+2
    assert np.isclose(metrics.mae(yt, yp), 0.75)
    assert np.isclose(metrics.rmse(yt, yp), np.sqrt((1 + 4) / 4))
    assert np.isclose(metrics.bias(yt, yp), 0.75)   # over-prediction


def test_metrics_are_nan_safe():
    yt = pd.Series([1.0, np.nan, 3.0])
    yp = pd.Series([1.0, 5.0, np.nan])
    assert np.isclose(metrics.mae(yt, yp), 0.0)     # only first pair survives


def test_skill_sign_convention():
    assert metrics.skill(0.5, 1.0) == 0.5           # half the baseline error
    assert metrics.skill(1.0, 1.0) == 0.0           # ties baseline
    assert metrics.skill(2.0, 1.0) == -1.0          # worse than baseline
    assert np.isnan(metrics.skill(1.0, 0.0))


def test_mase_against_naive_scale():
    insample = pd.Series(np.arange(10, dtype=float))  # naive 1-step MAE = 1
    yt = pd.Series([1.0, 2.0, 3.0])
    yp = pd.Series([1.5, 2.5, 3.5])                    # MAE 0.5
    assert np.isclose(metrics.mase(yt, yp, insample, m=1), 0.5)


def test_wape_and_smape_bounded():
    yt = pd.Series([10.0, 20.0, 30.0])
    yp = pd.Series([11.0, 19.0, 33.0])
    assert np.isclose(metrics.wape(yt, yp), (1 + 1 + 3) / 60)
    assert 0 <= metrics.smape(yt, yp) <= 200
