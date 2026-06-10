"""EDA — wave network. Generates the wave coverage matrix and the Mooloolaba
target distribution, decomposition, ACF/PACF, and seasonality calendar.

    ./.venv/bin/python notebooks/eda_wave.py
"""
import matplotlib
matplotlib.use("Agg")

import pandas as pd

import forecast as fc
import viz
from forecast.constants import FIGURE_DIR, STEPS_PER_YEAR

viz.apply_style()


def main() -> None:
    sources = fc.available_sources()["wave"]
    print("wave sources:", sources)

    # Fig 2.1 — coverage across the whole buoy network
    cov = pd.DataFrame({b: fc.load_wave(b)["hsig_m"] for b in sources})
    viz.save(viz.coverage_matrix(cov, freq="MS"), FIGURE_DIR / "wave_coverage.png")
    print("wrote wave_coverage.png")

    y = fc.load_target()
    print(f"target rows={len(y)} NaN%={y.isna().mean():.3%} "
          f"span={y.index.min().date()}..{y.index.max().date()}")

    # Fig 2.2 — distribution raw/log/Yeo-Johnson
    viz.save(viz.target_distribution(y), FIGURE_DIR / "target_distribution.png")
    # Fig 2.3 — STL + stationarity + spectral
    viz.save(viz.decomposition(y, period=365), FIGURE_DIR / "decomposition.png")
    # Fig 2.4 — ACF/PACF (target + companion period channel)
    wave = fc.load_wave("mooloolaba")
    viz.save(
        viz.acf_pacf({"hsig_m": y, "tz_s": wave["tz_s"], "hmax_m": wave["hmax_m"]},
                     max_lag=192),
        FIGURE_DIR / "acf_pacf.png",
    )
    # Fig 2.5 — seasonality calendar
    viz.save(viz.seasonality_calendar(y), FIGURE_DIR / "seasonality_calendar.png")
    print("wrote distribution / decomposition / acf_pacf / seasonality_calendar")


if __name__ == "__main__":
    main()
