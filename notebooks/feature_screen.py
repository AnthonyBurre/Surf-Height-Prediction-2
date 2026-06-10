"""EDA — feature × horizon predictive screen (Fig 2.7 / Phase-7 filter rung).
Mutual information between each engineered feature(t) and the target(t+h), per
horizon. The shortlist and the horizon-dependence of importance fall out of this.

    ./.venv/bin/python notebooks/feature_screen.py
"""
import matplotlib
matplotlib.use("Agg")

import forecast as fc
import viz
from forecast.constants import FIGURE_DIR, HORIZONS_H

viz.apply_style()


def main() -> None:
    y = fc.load_target()
    ds = fc.build_dataset(buoys=("mooloolaba",))
    X = fc.build_feature_matrix(ds, value_cols=fc.target_value_cols(ds))
    y_by_h = {h: fc.make_target(y, h) for h in HORIZONS_H}

    fig = viz.feature_horizon_screen(X, y_by_h, method="mutual_info", top=25)
    viz.save(fig, FIGURE_DIR / "feature_horizon_screen.png")
    print(f"wrote feature_horizon_screen.png  (X={X.shape}, horizons={list(HORIZONS_H)})")


if __name__ == "__main__":
    main()
