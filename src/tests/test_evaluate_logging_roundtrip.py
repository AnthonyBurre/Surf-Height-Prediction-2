import json

import pandas as pd

import forecast as fc
from forecast.evaluate import EvalResult


def _small_eval(y):
    spl = fc.RollingOriginSplitter(n_folds=2, val_size=1500, embargo_steps=12)
    return fc.evaluate(
        lambda h, s: fc.RidgeForecaster(alpha=1.0),
        y_full=y, X=None, splitter=spl, horizons=(6,),
        name="ridge_test", n_boot=100, data_sources=["mooloolaba"],
    )


def test_evaluate_produces_wellformed_result(synthetic_series):
    res = _small_eval(synthetic_series)
    assert isinstance(res, EvalResult)
    assert 6 in res.metrics and "rmse" in res.metrics[6]
    assert 6 in res.skill and 6 in res.paired
    recs = res.to_records()
    assert len(recs) == 1
    # JSON-serializable
    json.dumps(recs[0], default=str)
    assert "predictions" not in recs[0]


def test_evaluate_and_log_appends_and_reads_back(synthetic_series, tmp_log):
    res = _small_eval(synthetic_series)
    fc.log_run(res, log_path=tmp_log)
    df = fc.read_log(tmp_log)
    assert len(df) == 1
    assert set(["name", "horizon_h", "rmse", "skill_rmse", "paired_sig"]).issubset(df.columns)
    assert df.iloc[0]["name"] == "ridge_test"


def test_log_run_accepts_dict_and_list(tmp_log):
    fc.log_run({"name": "external", "horizon_h": 24, "rmse": 0.5}, log_path=tmp_log)
    fc.log_run([{"name": "external", "horizon_h": 48, "rmse": 0.6}], log_path=tmp_log)
    df = fc.read_log(tmp_log)
    assert len(df) == 2
    assert set(df["horizon_h"]) == {24, 48}


def test_read_log_empty_is_empty_frame(tmp_path):
    df = fc.read_log(tmp_path / "nope.jsonl")
    assert isinstance(df, pd.DataFrame) and len(df) == 0
