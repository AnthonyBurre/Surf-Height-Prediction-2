from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from qld_ckan.wave.downloader import _normalize_columns, _session, fetch_all, fetch_year_datastore


# ---------------------------------------------------------------------------
# _normalize_columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "year, input_cols, expected_cols",
    [
        # pre-2017: Dir_Tp TRUE → Peak Direction
        (2015,
         ["Date/Time", "Hs", "Dir_Tp TRUE", "SST"],
         ["Date/Time", "Hs", "Peak Direction", "SST"]),
        # boundary: 2016 still gets the pre-2017 rename
        (2016,
         ["Date/Time", "Hs", "Dir_Tp TRUE", "SST"],
         ["Date/Time", "Hs", "Peak Direction", "SST"]),
        # 2017+: Dir_Tp TRUE is left alone (it shouldn't appear, but if it
        # somehow does, don't rename — the canonical name applies only to
        # the pre-2017 era).
        (2017,
         ["Date/Time", "Dir_Tp TRUE"],
         ["Date/Time", "Dir_Tp TRUE"]),
        # mid-era: already-canonical columns pass through unchanged
        (2017,
         ["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Peak Direction", "SST"],
         ["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Peak Direction", "SST"]),
        # unit suffixes get stripped
        (2022,
         ["Date/Time (AEST)", "Hs (m)", "Hmax (m)", "Peak Direction (degrees)", "SST (degrees C)"],
         ["Date/Time", "Hs", "Hmax", "Peak Direction", "SST"]),
        # noop when no suffixes are present
        (2019,
         ["Date/Time", "Hs", "Peak Direction"],
         ["Date/Time", "Hs", "Peak Direction"]),
    ],
)
def test_normalize_columns(year, input_cols, expected_cols):
    df = pd.DataFrame(columns=input_cols)
    result = _normalize_columns(df, year)
    assert list(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# fetch_year_datastore
# ---------------------------------------------------------------------------


def test_fetch_year_datastore_returns_dataframe(datastore_response, wave_records_mid_era):
    with patch.object(_session(), "get", return_value=datastore_response(wave_records_mid_era)):
        df = fetch_year_datastore(2017, "fake-rid")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_fetch_year_datastore_drops_id_column(datastore_response, wave_records_mid_era):
    with patch.object(_session(), "get", return_value=datastore_response(wave_records_mid_era)):
        df = fetch_year_datastore(2017, "fake-rid")
    assert "_id" not in df.columns


def test_fetch_year_datastore_parses_datetime(datastore_response, wave_records_mid_era):
    with patch.object(_session(), "get", return_value=datastore_response(wave_records_mid_era)):
        df = fetch_year_datastore(2017, "fake-rid")
    assert pd.api.types.is_datetime64_any_dtype(df["Date/Time"])


def test_fetch_year_datastore_applies_pre_2017_normalisation(datastore_response, wave_records_pre_2017):
    with patch.object(_session(), "get", return_value=datastore_response(wave_records_pre_2017)):
        df = fetch_year_datastore(2015, "fake-rid")
    assert "Peak Direction" in df.columns
    assert "Dir_Tp TRUE" not in df.columns


def test_fetch_year_datastore_strips_unit_suffixes(datastore_response, wave_records_with_units):
    with patch.object(_session(), "get", return_value=datastore_response(wave_records_with_units)):
        df = fetch_year_datastore(2022, "fake-rid")
    assert "Hs" in df.columns
    assert "Hs (m)" not in df.columns
    assert "Date/Time" in df.columns


def test_fetch_year_datastore_paginates_when_total_exceeds_batch(datastore_response, wave_records_mid_era):
    page1 = wave_records_mid_era
    page2 = [{"_id": 3, "Date/Time": "2017-01-01T01:00:00", "Hs": 1.20, "Hmax": 2.00,
              "Tz": 5.60, "Tp": 9.20, "Peak Direction": 100.0, "SST": 25.2}]
    total = len(page1) + len(page2)

    with patch.object(_session(), "get", side_effect=[
        datastore_response(page1, total=total),
        datastore_response(page2, total=total),
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
    with patch("qld_ckan.wave.downloader.fetch_year_datastore", return_value=pd.DataFrame()) as mock_ds:
        frames = fetch_all(resource_ids=resource_ids)
    assert mock_ds.call_count == 2
    assert len(frames) == 2


@pytest.mark.parametrize(
    "status_code, expected_frames, raises",
    [
        (404, 0, False),  # silently skipped
        (500, None, True),  # re-raised
    ],
)
def test_fetch_all_handles_http_errors(status_code, expected_frames, raises):
    error = requests.exceptions.HTTPError(response=MagicMock(status_code=status_code))
    with patch("qld_ckan.wave.downloader.fetch_year_datastore", side_effect=error):
        if raises:
            with pytest.raises(requests.exceptions.HTTPError):
                fetch_all(resource_ids={2017: "rid"})
        else:
            frames = fetch_all(resource_ids={2017: "rid"})
            assert len(frames) == expected_frames
