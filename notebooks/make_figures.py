"""Rebuild the Phase-11 result figures from ``experiments.jsonl``.

``--plot-only`` rebuilds everything derivable from the log alone (money chart,
forest plots, source-ablation heatmap) without re-running the sweep. Without it,
the residual-diagnostics and feature-importance panels are also rebuilt, which
needs a few quick model refits (predictions/coefficients aren't stored in the
log).

    ./.venv/bin/python notebooks/make_figures.py [--plot-only]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import forecast as fc
import viz
from forecast.constants import FIGURE_DIR, HORIZON_STEPS, HORIZONS_H

EMBARGO = HORIZON_STEPS[max(HORIZONS_H)]
SELECT_MODELS = ["seasonal_mean", "ridge_primary", "hgb_primary"]
FAMILIES = ["hsig_m", "hmax_m", "tz_s", "tp_s", "peak_dir", "calendar"]


def _latest(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["name", "horizon_h", "mode"], keep="last")


def money_and_forest(df):
    dev = df[df["mode"].isin(["baseline", "select"])]   # exclude confirm/nn_screen rows
    nn = [n for n in dev[dev["mode"] == "select"]["name"].unique() if n.startswith("nn_")]
    models = SELECT_MODELS + nn
    keep = dev[dev["name"].isin(["persistence", *models])]
    fig = viz.skill_vs_horizon(keep, models=models, baseline_name="persistence", mode=None,
                               title="Error & skill vs lead time (dev, rolling-origin)")
    viz.save(fig, FIGURE_DIR / "skill_vs_horizon.png")
    print("wrote skill_vs_horizon.png")

    contenders = dev
    for h in (6, 24, 72):
        viz.save(viz.forest_plot(contenders, h, mode=None), FIGURE_DIR / f"forest_{h}h.png")
    print("wrote forest_{6,24,72}h.png")


def ablation(df):
    abl = df[df["mode"] == "ablation"]
    if abl.empty:
        print("no ablation rows — skipping ablation heatmap")
        return
    piv = abl.pivot_table("rmse", "name", "horizon_h")
    hs = sorted(piv.columns)
    add = pd.DataFrame(index=["wind", "neighbours"], columns=hs, dtype=float)
    drop = pd.DataFrame(index=["wind", "neighbours"], columns=hs, dtype=float)
    for h in hs:
        p, w, n, a = (piv.loc[k, h] for k in ["abl_primary", "abl_wind", "abl_neighbours", "abl_all"])
        add.loc["wind", h] = p - w            # add wind to primary (RMSE drop = help)
        add.loc["neighbours", h] = p - n
        drop.loc["wind", h] = n - a            # remove wind from all (given neighbours)
        drop.loc["neighbours", h] = w - a
    fig = viz.importance_horizon(add, drop_one=drop,
                                 titles=("add-one gain (RMSE↓)", "drop-one cost (RMSE↓)"))
    viz.save(fig, FIGURE_DIR / "source_ablation.png")
    print("wrote source_ablation.png")


def residual_and_importance():
    y = fc.load_target()
    dev, _ = fc.blind_split(y.index, embargo_steps=EMBARGO)
    ds = fc.build_dataset(buoys=(fc.TARGET_BUOY,))
    X = fc.build_feature_matrix(ds, value_cols=fc.target_value_cols(ds))

    # residual diagnostics for ridge at h=24 (one rolling-origin pass)
    spl = fc.RollingOriginSplitter(3, 5760, embargo_steps=EMBARGO)
    yh = fc.make_target(y, 24)
    Xa, ya = fc.align_xy(X, yh)
    Xa, ya = Xa.loc[Xa.index.isin(dev)], ya.loc[ya.index.isin(dev)]
    from forecast.backtest import rolling_origin
    rr = rolling_origin(lambda s: fc.RidgeForecaster(alpha=10.0), Xa, ya, spl)
    feat = y.reindex(rr.predictions.index)
    fig = viz.residual_diagnostics(rr.predictions, feature=feat, feature_name="current hsig_m (m)")
    viz.save(fig, FIGURE_DIR / "residual_diagnostics.png")
    print("wrote residual_diagnostics.png")

    # ridge coefficient importance grouped by feature family, per horizon
    imp = pd.DataFrame(index=FAMILIES, columns=list(HORIZONS_H), dtype=float)
    train = dev[:-1]
    for h in HORIZONS_H:
        yh = fc.make_target(y, h)
        Xa, ya = fc.align_xy(X, yh)
        Xa, ya = Xa.loc[Xa.index.isin(dev)], ya.loc[ya.index.isin(dev)]
        m = fc.RidgeForecaster(alpha=10.0).fit(Xa, ya)
        coef = np.abs(m.pipe_.named_steps["est"].coef_)
        s = pd.Series(coef, index=Xa.columns)
        for fam in FAMILIES:
            if fam == "calendar":
                mask = s.index.str.startswith(("hour_", "doy_"))
            elif fam == "peak_dir":
                mask = s.index.str.startswith("peak_dir_deg")
            else:
                mask = s.index.str.startswith(fam)
            imp.loc[fam, h] = s[mask].sum()
    imp = imp.div(imp.sum(axis=0), axis=1)  # share of |coef| per horizon
    fig = viz.importance_horizon(imp, titles=("Ridge |coef| share by feature family",))
    viz.save(fig, FIGURE_DIR / "importance_family.png")
    print("wrote importance_family.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="rebuild only log-derived figures (no model refits)")
    args = ap.parse_args()
    viz.apply_style()

    df = _latest(fc.read_log())
    if df.empty:
        print("experiments.jsonl is empty — run baselines.py / select_backtest.py first.")
        return
    money_and_forest(df)
    ablation(df)
    if not args.plot_only:
        residual_and_importance()


if __name__ == "__main__":
    main()
