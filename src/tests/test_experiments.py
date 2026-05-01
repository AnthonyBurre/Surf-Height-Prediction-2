import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from forecast import (
    EvaluationResult,
    PersistenceForecaster,
    evaluate,
    evaluate_and_log,
    log_run,
    read_log,
)
from forecast.experiments import _jsonable, _model_hyperparams


# ---------------------------------------------------------------------------
# _jsonable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(7), 7),
        (np.float32(1.5), pytest.approx(1.5)),
        (np.float64("nan"), None),
        (np.array([1, 2, 3]), [1, 2, 3]),
        (Path("/tmp/x"), "/tmp/x"),
        ({"a": np.int64(1), "b": [np.float64(2.0), "x"]}, {"a": 1, "b": [2.0, "x"]}),
    ],
    ids=["np_int", "np_float32", "np_nan", "np_array", "path", "nested"],
)
def test_jsonable(value, expected):
    assert _jsonable(value) == expected


def test_jsonable_handles_timestamp():
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    assert _jsonable(ts).startswith("2020-01-01")


def test_jsonable_falls_back_to_str_for_unknown_types():
    class Weird:
        def __repr__(self):
            return "<Weird>"
    assert _jsonable(Weird()) == "<Weird>"


# ---------------------------------------------------------------------------
# _model_hyperparams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected_subset",
    [
        (Ridge(alpha=2.5, fit_intercept=False), {"alpha": 2.5, "fit_intercept": False}),
        (PersistenceForecaster(target_col="custom"), {"target_col": "custom"}),
        (None, {}),
    ],
    ids=["sklearn_get_params", "custom_vars", "none"],
)
def test_model_hyperparams(model, expected_subset):
    params = _model_hyperparams(model)
    if not expected_subset:
        assert params == {}
    else:
        for key, val in expected_subset.items():
            assert params[key] == val


# ---------------------------------------------------------------------------
# log_run
# ---------------------------------------------------------------------------


def test_log_run_appends_one_record(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="persistence")
    log_path = tmp_path / "exp.jsonl"

    log_run(
        result,
        data_sources=["mooloolaba"],
        train_index=Xtr.index,
        test_index=Xte.index,
        n_features=Xtr.shape[1],
        path=log_path,
    )

    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["name"] == "persistence"
    assert parsed["model_class"] == "PersistenceForecaster"
    assert parsed["data_sources"] == ["mooloolaba"]
    assert parsed["train"]["n"] == len(Xtr)
    assert parsed["test"]["n"] == len(Xte)
    assert parsed["n_features"] == Xtr.shape[1]
    assert "MAE" in parsed["metrics"]


def test_log_run_appends_does_not_overwrite(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    for n in ("a", "b", "c"):
        result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name=n)
        log_run(result, data_sources=["s"],
                train_index=Xtr.index, test_index=Xte.index, path=log_path)
    assert len(log_path.read_text().splitlines()) == 3


def test_log_run_records_git_sha_when_in_repo(tmp_path, split):
    """Test suite runs from a git repo, so sha should be a 40-char hex string
    (possibly with -dirty suffix). Off-repo runs should yield None."""
    _df, Xtr, Xte, ytr, yte = split
    result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="r")
    record = log_run(result, data_sources=["s"],
                     train_index=Xtr.index, test_index=Xte.index, path=tmp_path / "x.jsonl")
    sha = record["git_sha"]
    assert sha is None or (len(sha.split("-")[0]) == 40)


def test_log_run_serialises_sklearn_hyperparams(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    result = evaluate(Ridge(alpha=0.7), Xtr, ytr, Xte, yte, name="ridge")
    log_run(result, data_sources=["s"],
            train_index=Xtr.index, test_index=Xte.index, path=log_path)
    parsed = json.loads(log_path.read_text())
    assert parsed["model_class"] == "Ridge"
    assert parsed["hyperparams"]["alpha"] == 0.7


def test_log_run_extra_field_passes_through(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    result = evaluate(PersistenceForecaster(), Xtr, ytr, Xte, yte, name="r")
    record = log_run(
        result, data_sources=["s"],
        train_index=Xtr.index, test_index=Xte.index,
        extra={"feature_set": "raw_v1", "notes": "smoke test"},
        path=tmp_path / "x.jsonl",
    )
    assert record["extra"]["feature_set"] == "raw_v1"


# ---------------------------------------------------------------------------
# evaluate_and_log
# ---------------------------------------------------------------------------


def test_evaluate_and_log_writes_and_returns_result(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    result = evaluate_and_log(
        PersistenceForecaster(), Xtr, ytr, Xte, yte,
        data_sources=["x"], name="run1", path=log_path,
    )
    assert isinstance(result, EvaluationResult)
    parsed = json.loads(log_path.read_text())
    assert parsed["name"] == "run1"
    assert parsed["metrics"]["RMSE"] == pytest.approx(result.metrics["RMSE"])


# ---------------------------------------------------------------------------
# read_log
# ---------------------------------------------------------------------------


def test_read_log_returns_dataframe(tmp_path, split):
    _df, Xtr, Xte, ytr, yte = split
    log_path = tmp_path / "exp.jsonl"
    for name in ["a", "b", "c"]:
        evaluate_and_log(PersistenceForecaster(), Xtr, ytr, Xte, yte,
                         data_sources=["s"], name=name, path=log_path)
    df_log = read_log(log_path)
    assert len(df_log) == 3
    assert {"timestamp", "name", "metrics"} <= set(df_log.columns)
    assert set(df_log["name"]) == {"a", "b", "c"}


def test_read_log_missing_file_returns_empty(tmp_path):
    assert read_log(tmp_path / "no-such-file.jsonl").empty


def test_read_log_empty_file_returns_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.touch()
    assert read_log(p).empty
