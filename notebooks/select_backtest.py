"""Exploratory model selection on the dev set (2015–2024) — the canonical
*exploratory* script (kept physically separate from the confirmatory confirm.py).

Climbs the architecture ladder (regularised linear -> gradient-boosted trees ->
sequence NN), all DIRECT per horizon, scored by rolling-origin with a horizon
embargo and a paired bootstrap vs persistence. A grouped source ablation
(primary / +wind / +neighbours / all) on identical common-overlap rows measures
whether the neighbour buoys or wind pay their way. Every run is appended to
``experiments.jsonl``. **The 2025 blind slice is never touched here.**

    ./.venv/bin/python notebooks/select_backtest.py --section all
    ./.venv/bin/python notebooks/select_backtest.py --section nn      # the long pole
"""
import argparse

import pandas as pd

import forecast as fc
from forecast.constants import HORIZON_STEPS, HORIZONS_H, NEIGHBOUR_BUOYS

N_FOLDS = 5
VAL_SIZE = 5760                              # ~120 days
EMBARGO = HORIZON_STEPS[max(HORIZONS_H)]     # 144 steps (3 d): safe for every horizon
PRIMARY_CHANNELS = ["hsig_m", "hmax_m", "tz_s", "tp_s"]   # NN raw channels

# NN sweep budget (CPU-bound; tuned for a multi-hour but finite run)
NN_FOLDS = 3
NN_SEEDS = (0, 1)
NN_STRIDE = 4
NN_MAX_CONTEXT = 192      # cap context (72h's 336-step window is too slow on CPU)
NN_EPOCHS = 18


def _setup():
    y = fc.load_target()
    dev, _ = fc.blind_split(y.index, embargo_steps=EMBARGO)
    splitter = fc.RollingOriginSplitter(N_FOLDS, VAL_SIZE, embargo_steps=EMBARGO)
    return y, dev, splitter


def _primary_X(y):
    ds = fc.build_dataset(buoys=(fc.TARGET_BUOY,))
    return fc.build_feature_matrix(ds, value_cols=fc.target_value_cols(ds))


def _report(res, name):
    line = "  ".join(
        f"{h}h={res.metrics[h]['rmse']:.3f}(sk{res.skill[h]['rmse']:+.2f}"
        f"{'*' if res.paired[h]['significant'] else ' '})"
        for h in HORIZONS_H
    )
    print(f"{name:20s} {line}")


# --------------------------------------------------------------------------- #
def _linear(estimator, name):
    # selection="random" + a looser tol keeps coordinate descent fast on the
    # highly-correlated lag features (full max_iter convergence is needless here).
    return fc.SklearnForecaster(estimator, name)


def run_linear(y, dev, splitter):
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    X = _primary_X(y)
    specs = {
        "ridge_primary": lambda h, s: fc.RidgeForecaster(alpha=10.0),
        "lasso_primary": lambda h, s: _linear(
            Lasso(alpha=5e-3, max_iter=3000, tol=1e-3, selection="random", random_state=0),
            "lasso"),
        "elasticnet_primary": lambda h, s: _linear(
            ElasticNet(alpha=5e-3, l1_ratio=0.5, max_iter=3000, tol=1e-3,
                       selection="random", random_state=0), "elasticnet"),
    }
    for name, fac in specs.items():
        res = fc.evaluate_and_log(fac, y_full=y, X=X, splitter=splitter, horizons=HORIZONS_H,
                                  index=dev, name=name, mode="select", n_boot=500,
                                  data_sources=["mooloolaba"])
        _report(res, name)


def run_trees(y, dev, splitter):
    X = _primary_X(y)
    res = fc.evaluate_and_log(
        lambda h, s: fc.HGBForecaster(max_iter=300, learning_rate=0.06, min_samples_leaf=200,
                                      random_state=s),
        y_full=y, X=X, splitter=splitter, horizons=HORIZONS_H, index=dev,
        name="hgb_primary", mode="select", n_boot=500, data_sources=["mooloolaba"])
    _report(res, "hgb_primary")


# --------------------------------------------------------------------------- #
def _cols_for(X, sources):
    keep = []
    for c in X.columns:
        if "__" in c:
            if c.split("__")[0] in sources:
                keep.append(c)
        else:
            keep.append(c)   # primary + calendar always present
    return keep


def run_ablation(y, dev, splitter):
    neighbours = [b for b in NEIGHBOUR_BUOYS if b in fc.available_sources()["wave"]]
    winds = fc.available_sources()["wind"]
    ds = fc.build_dataset(buoys=(fc.TARGET_BUOY, *neighbours), stations=winds)
    rep = (["hsig_m"] + [f"{b}__hsig_m" for b in neighbours]
           + [f"{s}__wind_speed_ms" for s in winds])
    start, end = fc.common_overlap(ds, rep)
    overlap = dev[(dev >= start) & (dev <= end)]
    print(f"ablation overlap rows: {len(overlap)}  ({start.date()}..{end.date()})")

    # reduced, uniform feature grid so the multi-source matrix stays tractable
    X = fc.build_feature_matrix(ds, lags=(1, 6, 48), windows=(48,), deltas=(6,),
                                roll_stats=("mean", "std"))
    sets = {
        "abl_primary": set(),
        "abl_wind": set(winds),
        "abl_neighbours": set(neighbours),
        "abl_all": set(neighbours) | set(winds),
    }
    for name, srcs in sets.items():
        Xs = X[_cols_for(X, srcs)]
        res = fc.evaluate_and_log(
            lambda h, s: fc.RidgeForecaster(alpha=10.0),
            y_full=y, X=Xs, splitter=splitter, horizons=HORIZONS_H, index=overlap,
            name=name, mode="ablation", n_boot=500, data_sources=sorted(srcs) or ["mooloolaba"])
        _report(res, f"{name} (p={Xs.shape[1]})")


# --------------------------------------------------------------------------- #
def run_nn(y, dev):
    from forecast import neural
    ds = fc.build_dataset(buoys=(fc.TARGET_BUOY,))

    def make_seq(arch, h, s):
        ctx = min(neural.context_for_horizon(h), NN_MAX_CONTEXT)
        return neural.SeqForecaster(ds, PRIMARY_CHANNELS, context_len=ctx, horizon_h=h,
                                    arch=arch, hidden=48, layers=2, dropout=0.1,
                                    epochs=NN_EPOCHS, train_stride=NN_STRIDE, seed=s)

    # 1) arch screen at representative horizons (short/mid/long), 1 fold, 1 seed
    screen_spl = fc.RollingOriginSplitter(1, VAL_SIZE, embargo_steps=EMBARGO)
    best = {}
    for arch in ("gru", "lstm", "tcn"):
        res = fc.evaluate_and_log(
            lambda h, s, a=arch: make_seq(a, h, s), y_full=y, X=None, splitter=screen_spl,
            horizons=(6, 24, 72), index=dev, name=f"nn_{arch}_screen", mode="nn_screen",
            seeds=(0,), n_boot=300, data_sources=["mooloolaba"])
        best[arch] = sum(res.metrics[h]["rmse"] for h in (6, 24, 72) if h in res.metrics)
        _report_partial(res, f"nn_{arch}_screen", (6, 24, 72))
    winner = min(best, key=best.get)
    print(f"NN arch winner: {winner}")

    # 2) best arch across all horizons, seed-averaged
    nn_spl = fc.RollingOriginSplitter(NN_FOLDS, VAL_SIZE, embargo_steps=EMBARGO)
    res = fc.evaluate_and_log(
        lambda h, s: make_seq(winner, h, s),
        y_full=y, X=None, splitter=nn_spl, horizons=HORIZONS_H, index=dev,
        name=f"nn_{winner}", mode="select", seeds=NN_SEEDS, n_boot=500,
        data_sources=["mooloolaba"],
        hyperparams={"arch": winner, "hidden": 48, "layers": 2, "stride": NN_STRIDE})
    _report(res, f"nn_{winner}")


def _report_partial(res, name, horizons):
    line = "  ".join(f"{h}h={res.metrics[h]['rmse']:.3f}" for h in horizons if h in res.metrics)
    print(f"{name:20s} {line}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all",
                    choices=["all", "linear", "trees", "ablation", "nn", "fast"])
    args = ap.parse_args()
    y, dev, splitter = _setup()
    print(f"dev: {dev.min().date()}..{dev.max().date()} ({len(dev)} origins)")

    if args.section in ("all", "fast", "linear"):
        run_linear(y, dev, splitter)
    if args.section in ("all", "fast", "trees"):
        run_trees(y, dev, splitter)
    if args.section in ("all", "fast", "ablation"):
        run_ablation(y, dev, splitter)
    if args.section in ("all", "nn"):
        run_nn(y, dev)


if __name__ == "__main__":
    main()
