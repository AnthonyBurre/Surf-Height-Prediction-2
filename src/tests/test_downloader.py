from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from wave_data.downloader import _normalize_columns, _session, fetch_all, fetch_year_datastore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


_RECORDS_MID_ERA = [
    {"_id": 1, "Date/Time": "2017-01-01T00:00:00", "Hs": 1.10, "Hmax": 1.90, "Tz": 5.50, "Tp": 9.00, "Peak Direction": 95.0, "SST": 25.0},
    {"_id": 2, "Date/Time": "2017-01-01T00:30:00", "Hs": 1.15, "Hmax": 1.95, "Tz": 5.55, "Tp": 9.10, "Peak Direction": 98.0, "SST": 25.1},
]

_RECORDS_PRE_2017 = [
    {"_id": 1, "Date/Time": "2015-01-01T00:00:00", "Hs": 1.14, "Hmax": 2.08, "Tz": 6.67, "Tp": 9.33, "Dir_Tp TRUE": 97.0, "SST": 26.95},
    {"_id": 2, "Date/Time": "2015-01-01T00:30:00", "Hs": 1.12, "Hmax": 1.92, "Tz": 6.37, "Tp": 10.12, "Dir_Tp TRUE": 92.0, "SST": 26.90},
]

_RECORDS_WITH_UNITS = [
    {"_id": 1, "Date/Time (AEST)": "2022-01-01T00:00:00", "Hs (m)": 1.20, "Hmax (m)": 2.00, "Tz (s)": 5.60, "Tp (s)": 9.50, "Peak Direction (degrees)": 100.0, "SST (degrees C)": 26.0},
    {"_id": 2, "Date/Time (AEST)": "2022-01-01T00:30:00", "Hs (m)": 1.25, "Hmax (m)": 2.10, "Tz (s)": 5.65, "Tp (s)": 9.55, "Peak Direction (degrees)": 102.0, "SST (degrees C)": 26.1},
]


# ---------------------------------------------------------------------------
# _normalize_columns
# ---------------------------------------------------------------------------


def test_normalize_pre_2017_renames_dir_tp_true():
    df = pd.DataFrame({"Date/Time": [], "Hs": [], "Dir_Tp TRUE": [], "SST": []})
    result = _normalize_columns(df, 2015)
    assert "Peak Direction" in result.columns
    assert "Dir_Tp TRUE" not in result.columns


def test_normalize_pre_2017_applied_to_2016():
    df = pd.DataFrame({"Date/Time": [], "Hs": [], "Dir_Tp TRUE": [], "SST": []})
    result = _normalize_columns(df, 2016)
    assert "Peak Direction" in result.columns


def test_normalize_mid_era_columns_unchanged():
    cols = ["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Peak Direction", "SST"]
    df = pd.DataFrame(columns=cols)
    result = _normalize_columns(df, 2017)
    assert list(result.columns) == cols


def test_normalize_strips_unit_suffixes():
    df = pd.DataFrame(columns=["Date/Time (AEST)", "Hs (m)", "Hmax (m)", "Peak Direction (degrees)", "SST (degrees C)"])
    result = _normalize_columns(df, 2022)
    assert list(result.columns) == ["Date/Time", "Hs", "Hmax", "Peak Direction", "SST"]


def test_normalize_strip_is_noop_when_no_suffixes():
    cols = ["Date/Time", "Hs", "Peak Direction"]
    df = pd.DataFrame(columns=cols)
    result = _normalize_columns(df, 2019)
    assert list(result.columns) == cols


def test_normalize_does_not_rename_dir_tp_true_for_2017_plus():
    df = pd.DataFrame({"Date/Time": [], "Dir_Tp TRUE": []})
    result = _normalize_columns(df, 2017)
    assert "Dir_Tp TRUE" in result.columns


# ---------------------------------------------------------------------------
# fetch_year_datastore
# ---------------------------------------------------------------------------


def test_fetch_year_datastore_returns_dataframe():
    with patch.object(_session(), "get", return_value=_datastore_response(_RECORDS_MID_ERA)):
        df = fetch_year_datastore(2017, "fake-rid")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_fetch_year_datastore_drops_id_column():
    with patch.object(_session(), "get", return_value=_datastore_response(_RECORDS_MID_ERA)):
        df = fetch_year_datastore(2017, "fake-rid")
    assert "_id" not in df.columns


def test_fetch_year_datastore_parses_datetime():
    with patch.object(_session(), "get", return_value=_datastore_response(_RECORDS_MID_ERA)):
        df = fetch_year_datastore(2017, "fake-rid")
    assert pd.api.types.is_datetime64_any_dtype(df["Date/Time"])


def test_fetch_year_datastore_applies_pre_2017_normalisation():
    with patch.object(_session(), "get", return_value=_datastore_response(_RECORDS_PRE_2017)):
        df = fetch_year_datastore(2015, "fake-rid")
    assert "Peak Direction" in df.columns
    assert "Dir_Tp TRUE" not in df.columns


def test_fetch_year_datastore_strips_unit_suffixes():
    with patch.object(_session(), "get", return_value=_datastore_response(_RECORDS_WITH_UNITS)):
        df = fetch_year_datastore(2022, "fake-rid")
    assert "Hs" in df.columns
    assert "Hs (m)" not in df.columns
    assert "Date/Time" in df.columns


def test_fetch_year_datastore_paginates_when_total_exceeds_batch():
    from wave_data.downloader import _DATASTORE_BATCH

    page1 = _RECORDS_MID_ERA
    page2 = [{"_id": 3, "Date/Time": "2017-01-01T01:00:00", "Hs": 1.20, "Hmax": 2.00,
               "Tz": 5.60, "Tp": 9.20, "Peak Direction": 100.0, "SST": 25.2}]
    total = len(page1) + len(page2)

    with patch.object(_session(), "get", side_effect=[
        _datastore_response(page1, total=total),
        _datastore_response(page2, total=total),
    ]):
        df = fetch_year_datastore(2017, "fake-rid")

    assert len(df) == total


def test_fetch_year_datastore_raises_on_http_error():
    mock = MagicMock()
    mock.raise_for_status.side_effect = requests.exceptions.HTTPError(response=MagicMock(status_code=500))
    with patch.object(_session(), "get", return_value=mock):
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_year_datastore(2017, "fake-rid")


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------


def test_fetch_all_returns_one_frame_per_resource_id():
    resource_ids = {2015: "rid-2015", 2017: "rid-2017"}
    with patch("wave_data.downloader.fetch_year_datastore", return_value=pd.DataFrame()) as mock_ds:
        frames = fetch_all(resource_ids=resource_ids)
    assert mock_ds.call_count == 2
    assert len(frames) == 2


def test_fetch_all_skips_404():
    resource_ids = {2017: "rid-2017", 2018: "rid-missing"}

    def side_effect(year, rid):
        if year == 2018:
            raise requests.exceptions.HTTPError(response=MagicMock(status_code=404))
        return pd.DataFrame(_RECORDS_MID_ERA)

    with patch("wave_data.downloader.fetch_year_datastore", side_effect=side_effect):
        frames = fetch_all(resource_ids=resource_ids)

    assert len(frames) == 1


def test_fetch_all_reraises_non_404_errors():
    resource_ids = {2017: "rid-2017"}
    with patch("wave_data.downloader.fetch_year_datastore",
               side_effect=requests.exceptions.HTTPError(response=MagicMock(status_code=500))):
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_all(resource_ids=resource_ids)


def test_fetch_all_returns_empty_list_when_all_404():
    resource_ids = {2017: "rid-missing"}
    with patch("wave_data.downloader.fetch_year_datastore",
               side_effect=requests.exceptions.HTTPError(response=MagicMock(status_code=404))):
        frames = fetch_all(resource_ids=resource_ids)
    assert frames == []
