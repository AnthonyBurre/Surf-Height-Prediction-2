"""EDA — seasonality deep-dive (Fig 2.5). Confirms the annual storm-season swing
and the (weak) diurnal cycle that motivate calendar features and the climatology
baseline.

    ./.venv/bin/python notebooks/seasonality.py
"""
import matplotlib
matplotlib.use("Agg")

import forecast as fc
import viz
from forecast.constants import FIGURE_DIR

viz.apply_style()


def main() -> None:
    y = fc.load_target().dropna()
    viz.save(viz.seasonality_calendar(y), FIGURE_DIR / "seasonality_calendar.png")

    by_month = y.groupby(y.index.month).mean()
    by_hour = y.groupby(y.index.hour).mean()
    print("mean hsig_m by month (storm season Dec–Mar):")
    print(by_month.round(2).to_string())
    print(f"\ndiurnal swing: {by_hour.max() - by_hour.min():.3f} m "
          f"vs annual swing: {by_month.max() - by_month.min():.3f} m")
    print("=> annual cycle dominates; diurnal is weak (drives month features, "
          "month×hour climatology baseline).")


if __name__ == "__main__":
    main()
