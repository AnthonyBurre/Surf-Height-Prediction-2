"""EDA — wind network. Generates the wind coverage matrix and the wind-speed
distribution / seasonality for the Mooloolaba-paired Mountain Creek station.

    ./.venv/bin/python notebooks/eda_wind.py
"""
import matplotlib
matplotlib.use("Agg")

import pandas as pd

import forecast as fc
import viz
from forecast.constants import FIGURE_DIR, WIND_PAIR

viz.apply_style()


def main() -> None:
    sources = fc.available_sources()["wind"]
    print("wind sources:", sources)

    cov = pd.DataFrame({s: fc.load_wind(s)["wind_speed_ms"] for s in sources})
    viz.save(viz.coverage_matrix(cov, freq="MS"), FIGURE_DIR / "wind_coverage.png")
    print("wrote wind_coverage.png")

    wind = fc.load_wind(WIND_PAIR)
    spd = wind["wind_speed_ms"].rename("wind_speed_ms")
    print(f"{WIND_PAIR}: rows={len(spd)} NaN%={spd.isna().mean():.3%}")
    viz.save(viz.target_distribution(spd), FIGURE_DIR / "wind_distribution.png")
    viz.save(viz.seasonality_calendar(spd), FIGURE_DIR / "wind_seasonality.png")
    print("wrote wind distribution / seasonality")


if __name__ == "__main__":
    main()
