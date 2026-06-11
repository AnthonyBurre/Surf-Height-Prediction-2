"""Phase-3 baselines on the dev set (2015–2024), via rolling-origin.

Scores persistence / seasonal-naive / climatology / drift at every horizon,
locates the persistence↔climatology crossover, and logs each to
``experiments.jsonl`` (mode="baseline"). Never touches the 2025 blind slice.

    ./.venv/bin/python notebooks/baselines.py
"""
import matplotlib
matplotlib.use("Agg")

import forecast as fc
import viz
from forecast.constants import FIGURE_DIR, HORIZON_STEPS, HORIZONS_H

N_FOLDS = 5
VAL_SIZE = 5760                       # ~120 days
EMBARGO = HORIZON_STEPS[max(HORIZONS_H)]  # 144 steps (3 d): safe for every horizon


def main() -> None:
    y = fc.load_target()
    dev, _ = fc.blind_split(y.index, embargo_steps=EMBARGO)
    splitter = fc.RollingOriginSplitter(N_FOLDS, VAL_SIZE, embargo_steps=EMBARGO)

    specs = {
        "persistence": fc.Persistence,
        "seasonal_naive": fc.SeasonalNaive,
        "seasonal_mean": fc.SeasonalMean,
        "drift": fc.DriftRandomWalk,
    }
    for name, ctor in specs.items():
        res = fc.evaluate_and_log(
            lambda h, s, c=ctor: c(y, h),
            y_full=y, X=None, splitter=splitter, horizons=HORIZONS_H,
            index=dev, name=name, mode="baseline", n_boot=500,
        )
        line = "  ".join(f"{h}h={res.metrics[h]['rmse']:.3f}(skill{res.skill[h]['rmse']:+.2f})"
                         for h in HORIZONS_H)
        print(f"{name:15s} {line}")

    df = fc.read_log()
    base = df[df["mode"] == "baseline"]
    fig = viz.skill_vs_horizon(base, mode="baseline", baseline_name="persistence",
                               title="Baselines & the persistence↔climatology crossover")
    viz.save(fig, FIGURE_DIR / "baseline_crossover.png")
    print("wrote baseline_crossover.png")


if __name__ == "__main__":
    main()
