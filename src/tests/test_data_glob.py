import numpy as np
import pandas as pd
import pytest

from forecast import data
from forecast.constants import SOURCE_TZ


def _write_wave(path, span="2015-2020", rows=200):
    idx = pd.date_range("2015-01-01", periods=rows, freq="30min", tz=SOURCE_TZ)
    df = pd.DataFrame({
        "hsig_m": np.linspace(1, 2, rows), "hmax_m": np.linspace(1.5, 3, rows),
        "tz_s": 6.0, "tp_s": 10.0, "peak_dir_deg": 90.0, "sst_c": 25.0,
    }, index=idx)
    df.index.name = "datetime"
    df.to_csv(path / f"mooloolaba_wave_data_{span}.csv")


def test_load_wave_parses_tz_index(tmp_path):
    _write_wave(tmp_path)
    df = data.load_wave("mooloolaba", data_dir=tmp_path)
    assert str(df.index.tz) == SOURCE_TZ
    assert list(df.columns) == ["hsig_m", "hmax_m", "tz_s", "tp_s", "peak_dir_deg", "sst_c"]


def test_available_sources_discovers_slugs(tmp_path):
    _write_wave(tmp_path)
    (tmp_path / "lytton_wind_data_2014-2024.csv").write_text(
        "datetime,wind_dir_deg,wind_speed_ms,wind_sigma_theta_deg,wind_speed_std_ms\n"
    )
    srcs = data.available_sources(tmp_path)
    assert srcs["wave"] == ["mooloolaba"]
    assert srcs["wind"] == ["lytton"]


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_wave("nonexistent", data_dir=tmp_path)


def test_multiple_matches_pick_widest_span(tmp_path):
    _write_wave(tmp_path, span="2015-2018")
    _write_wave(tmp_path, span="2015-2025")   # wider
    chosen = data._find_one("mooloolaba_wave_data_*.csv", tmp_path)
    assert chosen.name.endswith("2015-2025.csv")
