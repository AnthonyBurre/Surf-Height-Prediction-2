from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from qld_ckan import unify_frames
from qld_ckan.wave.pipeline import clean, run, unify


# ---------------------------------------------------------------------------
# qld_ckan.unify_frames — shared helper exercised through the wave wrapper
# elsewhere; covered directly here so future sources don't have to.
# ---------------------------------------------------------------------------


def test_unify_frames_concatenates_with_reset_index():
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"x": [3]})
    out = unify_frames([a, b])
    assert list(out["x"]) == [1, 2, 3]
    assert list(out.index) == [0, 1, 2]


def test_unify_frames_raises_on_empty_list():
    with pytest.raises(ValueError, match="No data was downloaded"):
        unify_frames([])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal raw DataFrame as fetch_year_datastore would return it."""
    return pd.DataFrame(rows)


def _standard_rows():
    return [
        {
            "Date/Time": pd.Timestamp("2017-01-01 00:00:00"),
            "Hs": 1.10, "Hmax": 1.90, "Tz": 5.50,
            "Tp": 9.00, "Peak Direction": 95.0, "SST": 25.0,
        },
        {
            "Date/Time": pd.Timestamp("2017-01-01 00:30:00"),
            "Hs": 1.15, "Hmax": 1.95, "Tz": 5.55,
            "Tp": 9.10, "Peak Direction": 98.0, "SST": 25.1,
        },
    ]


# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------


def test_clean_renames_all_columns():
    df = _raw_frame(_standard_rows())
    result = clean(df)
    assert set(result.columns) == {"hsig_m", "hmax_m", "tz_s", "tp_s", "peak_dir_deg", "sst_c"}


def test_clean_sets_datetime_index():
    df = _raw_frame(_standard_rows())
    result = clean(df)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "datetime_utc"


def test_clean_sorts_index():
    rows = list(reversed(_standard_rows()))
    df = _raw_frame(rows)
    result = clean(df)
    assert result.index.is_monotonic_increasing


def test_clean_drops_rows_with_nat_datetime():
    rows = _standard_rows()
    rows.append({
        "Date/Time": pd.NaT,
        "Hs": 1.0, "Hmax": 1.5, "Tz": 5.0,
        "Tp": 8.0, "Peak Direction": 90.0, "SST": 24.0,
    })
    df = _raw_frame(rows)
    result = clean(df)
    assert len(result) == 2
    assert result.index.isna().sum() == 0


def test_clean_preserves_row_count_when_no_bad_rows():
    df = _raw_frame(_standard_rows())
    result = clean(df)
    assert len(result) == 2


def _ts(s: str) -> pd.Timestamp:
    """Tz-aware UTC timestamp matching the cleaned DataFrame's index.

    The argument is interpreted as AEST (matching the raw source) and
    converted to UTC, since ``clean()`` localises the naive source then
    converts to UTC for storage.
    """
    return pd.Timestamp(s, tz="Australia/Brisbane").tz_convert("UTC")


def test_clean_preserves_values():
    df = _raw_frame(_standard_rows())
    result = clean(df)
    assert result.loc[_ts("2017-01-01 00:00:00"), "hsig_m"] == pytest.approx(1.10)


def test_clean_replaces_sentinel_with_nan():
    rows = _standard_rows()
    rows[0]["Hs"] = -99.9
    rows[1]["SST"] = -99.9
    df = _raw_frame(rows)
    result = clean(df)
    assert np.isnan(result.loc[_ts("2017-01-01 00:00:00"), "hsig_m"])
    assert np.isnan(result.loc[_ts("2017-01-01 00:30:00"), "sst_c"])
    # other values unchanged
    assert result.loc[_ts("2017-01-01 00:00:00"), "sst_c"] == pytest.approx(25.0)


def test_clean_coerces_string_numerics():
    rows = _standard_rows()
    # Simulate CKAN returning numerics as strings, including the sentinel
    rows[0]["Hs"] = "1.10"
    rows[1]["Hs"] = "-99.9"
    df = _raw_frame(rows)
    result = clean(df)
    assert result["hsig_m"].dtype.kind == "f"
    assert result.loc[_ts("2017-01-01 00:00:00"), "hsig_m"] == pytest.approx(1.10)
    assert np.isnan(result.loc[_ts("2017-01-01 00:30:00"), "hsig_m"])


def test_clean_index_is_tz_aware_utc():
    df = _raw_frame(_standard_rows())
    result = clean(df)
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"


def test_clean_reindexes_gaps_as_nan_rows():
    # Two timestamps 1.5 hours apart leave two missing 30-min slots between.
    rows = [
        {"Date/Time": pd.Timestamp("2017-01-01 00:00:00"), "Hs": 1.10, "Hmax": 1.9,
         "Tz": 5.5, "Tp": 9.0, "Peak Direction": 95.0, "SST": 25.0},
        {"Date/Time": pd.Timestamp("2017-01-01 01:30:00"), "Hs": 1.20, "Hmax": 2.0,
         "Tz": 5.6, "Tp": 9.1, "Peak Direction": 96.0, "SST": 25.1},
    ]
    result = clean(_raw_frame(rows))
    assert len(result) == 4
    assert np.isnan(result.loc[_ts("2017-01-01 00:30:00"), "hsig_m"])
    assert np.isnan(result.loc[_ts("2017-01-01 01:00:00"), "hsig_m"])


def test_clean_drops_duplicate_timestamps():
    rows = _standard_rows()
    # Duplicate the first row's timestamp with different values
    rows.append({
        "Date/Time": pd.Timestamp("2017-01-01 00:00:00"),
        "Hs": 9.99, "Hmax": 9.99, "Tz": 9.99,
        "Tp": 9.99, "Peak Direction": 9.99, "SST": 9.99,
    })
    result = clean(_raw_frame(rows))
    # Only 2 unique timestamps, and the first-seen value wins.
    assert len(result) == 2
    assert result.loc[_ts("2017-01-01 00:00:00"), "hsig_m"] == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# unify()
# ---------------------------------------------------------------------------


def test_unify_concatenates_frames_from_fetch_all():
    frame_a = _raw_frame(_standard_rows())
    frame_b = _raw_frame([{
        "Date/Time": pd.Timestamp("2018-01-01 00:00:00"),
        "Hs": 1.20, "Hmax": 2.00, "Tz": 5.60,
        "Tp": 9.50, "Peak Direction": 100.0, "SST": 26.0,
    }])

    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame_a, frame_b]):
        result = unify()

    assert len(result) == 3
    assert list(result.columns) == list(frame_a.columns)


def test_unify_resets_index():
    frame = _raw_frame(_standard_rows())
    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame, frame]):
        result = unify()
    assert list(result.index) == list(range(len(result)))


def test_unify_raises_when_no_data_downloaded():
    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[]):
        with pytest.raises(ValueError, match="No data was downloaded"):
            unify()


def test_unify_passes_custom_resource_ids_to_fetch_all():
    custom_ids = {2020: "fake-resource-id"}
    frame = _raw_frame(_standard_rows())

    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame]) as mock_fetch:
        unify(resource_ids=custom_ids)

    mock_fetch.assert_called_once_with(custom_ids)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_writes_csv_and_creates_parent_dir(tmp_path):
    output = tmp_path / "nested" / "out.csv"
    frame = _raw_frame(_standard_rows())

    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame]):
        result = run(output_path=output)

    assert output.exists()
    assert len(result) == 2

    reloaded = pd.read_csv(output, parse_dates=["datetime_utc"], index_col="datetime_utc")
    assert set(reloaded.columns) == set(result.columns)
    assert len(reloaded) == len(result)


def test_run_accepts_string_output_path(tmp_path):
    output = str(tmp_path / "out.csv")
    frame = _raw_frame(_standard_rows())

    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame]):
        run(output_path=output)

    assert (tmp_path / "out.csv").exists()
