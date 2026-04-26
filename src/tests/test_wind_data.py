from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from wind_data.downloader import fetch_all, fetch_year_datastore
from wind_data.pipeline import clean


def _datastore_response(records: list[dict], total: int | None = None) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "result": {
            "total": total if total is not None else len(records),
            "records": records,
        }
    }
    return mock


_RAW_RECORDS = [
    {
        "_id": 1,
        "Date": "2024-01-01T00:00:00",
        "Time": "00:00",
        "Wind Direction (degTN)": 180,
        "Wind Speed (m/s)": 2.5,
        "Wind Sigma Theta (deg)": 18.0,
        "Wind Speed Std Dev (m/s)": 0.7,
        "Ozone (ppm)": 0.02,
        "PM10 (ug/m^3)": 12.0,
    },
    {
        "_id": 2,
        "Date": "2024-01-01T00:00:00",
        "Time": "01:00",
        "Wind Direction (degTN)": 185,
        "Wind Speed (m/s)": 2.7,
        "Wind Sigma Theta (deg)": 17.0,
        "Wind Speed Std Dev (m/s)": 0.8,
        "Ozone (ppm)": 0.02,
        "PM10 (ug/m^3)": 11.5,
    },
]


# ---------------------------------------------------------------------------
# fetch_year_datastore
# ---------------------------------------------------------------------------


def test_fetch_year_drops_id_column():
    with patch("wind_data.downloader._session") as mock_session:
        mock_session.return_value.get.return_value = _datastore_response(_RAW_RECORDS)
        df = fetch_year_datastore(2024, "rid")
    assert "_id" not in df.columns
    assert "Wind Speed (m/s)" in df.columns


def test_fetch_year_paginates_until_total_reached():
    page1 = _datastore_response(_RAW_RECORDS[:1], total=2)
    page2 = _datastore_response(_RAW_RECORDS[1:], total=2)
    with patch("wind_data.downloader._session") as mock_session:
        mock_session.return_value.get.side_effect = [page1, page2]
        df = fetch_year_datastore(2024, "rid")
    assert len(df) == 2
    assert mock_session.return_value.get.call_count == 2


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


def test_fetch_all_skips_404():
    error_response = MagicMock(status_code=404)
    http_error = requests.exceptions.HTTPError(response=error_response)

    def _get(*_args, **_kwargs):
        mock = MagicMock()
        mock.raise_for_status.side_effect = http_error
        return mock

    with patch("wind_data.downloader._session") as mock_session:
        mock_session.return_value.get.side_effect = _get
        frames = fetch_all({2099: "missing"})
    assert frames == []


def test_fetch_all_reraises_non_404():
    error_response = MagicMock(status_code=500)
    http_error = requests.exceptions.HTTPError(response=error_response)

    def _get(*_args, **_kwargs):
        mock = MagicMock()
        mock.raise_for_status.side_effect = http_error
        return mock

    with patch("wind_data.downloader._session") as mock_session:
        mock_session.return_value.get.side_effect = _get
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_all({2024: "rid"})


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def test_clean_combines_date_and_time_into_utc_index():
    df = clean(pd.DataFrame(_RAW_RECORDS).drop(columns=["_id"]))
    # 2024-01-01 00:00 Brisbane (UTC+10) is 2023-12-31 14:00 UTC.
    assert df.index[0] == pd.Timestamp("2023-12-31 14:00", tz="UTC")
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "datetime_utc"


def test_clean_renames_wind_columns_and_drops_pollutants():
    df = clean(pd.DataFrame(_RAW_RECORDS).drop(columns=["_id"]))
    assert list(df.columns) == [
        "wind_dir_deg",
        "wind_speed_ms",
        "wind_sigma_theta_deg",
        "wind_speed_std_ms",
    ]


def test_clean_reindexes_onto_hourly_grid():
    # Drop the middle of three records; clean should fill the gap with NaN.
    records = [
        {"Date": "2024-06-01T00:00:00", "Time": "00:00", "Wind Speed (m/s)": 1.0, "Wind Direction (degTN)": 90},
        {"Date": "2024-06-01T00:00:00", "Time": "02:00", "Wind Speed (m/s)": 1.5, "Wind Direction (degTN)": 95},
    ]
    df = clean(pd.DataFrame(records))
    assert len(df) == 3  # 00:00, 01:00, 02:00 (Brisbane)
    assert df["wind_speed_ms"].isna().sum() == 1
    assert (df.index[1] - df.index[0]) == pd.Timedelta("1h")


def test_clean_drops_duplicate_timestamps():
    duplicated = _RAW_RECORDS + [_RAW_RECORDS[0]]
    df = clean(pd.DataFrame(duplicated).drop(columns=["_id"]))
    assert df.index.is_unique


def test_clean_coerces_string_numerics():
    records = [
        {"Date": "2024-06-01T00:00:00", "Time": "00:00", "Wind Speed (m/s)": "1.5", "Wind Direction (degTN)": "90"},
    ]
    df = clean(pd.DataFrame(records))
    assert df["wind_speed_ms"].dtype.kind == "f"
    assert df["wind_speed_ms"].iloc[0] == 1.5


def test_clean_tolerates_missing_optional_columns():
    # 2024 records lack Air Temperature; ensure clean handles a frame that
    # only carries the two core wind fields.
    minimal = [
        {"Date": "2024-06-01T00:00:00", "Time": "00:00", "Wind Speed (m/s)": 2.0, "Wind Direction (degTN)": 100},
    ]
    df = clean(pd.DataFrame(minimal))
    assert "wind_speed_ms" in df.columns
    assert "wind_sigma_theta_deg" not in df.columns  # not in input → not in output
