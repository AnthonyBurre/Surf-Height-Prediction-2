"""Confirmatory pass — score a pre-registered candidate set **once** on the 2025
blind slice (the canonical *confirmatory* script). Each candidate's pipeline is
re-fit on the dev set and scored on the never-touched blind period via a single
dev->blind fold. Logged with mode="confirm".

    ./.venv/bin/python notebooks/confirm.py [--nn-arch gru]

Run this only after select_backtest.py has frozen the recipe — and only once.
"""
import argparse

import forecast as fc
from forecast.constants import HORIZON_STEPS, HORIZONS_H

EMBARGO = HORIZON_STEPS[max(HORIZONS_H)]
PRIMARY_CHANNELS = ["hsig_m", "hmax_m", "tz_s", "tp_s"]


def _report(res, name):
    line = "  ".join(
        f"{h}h={res.metrics[h]['rmse']:.3f}(sk{res.skill[h]['rmse']:+.2f}"
        f"{'*' if res.paired[h]['significant'] else ' '})"
        for h in HORIZONS_H if h in res.metrics
    )
    print(f"{name:18s} {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-arch", default=None, help="include the sequence NN with this arch")
    args = ap.parse_args()

    y = fc.load_target()
    dev, blind = fc.blind_split(y.index, embargo_steps=EMBARGO)
    print(f"dev {dev.min().date()}..{dev.max().date()} | "
          f"BLIND {blind.min().date()}..{blind.max().date()} ({len(blind)} origins)")
    splitter = fc.FixedSplit(dev, blind)
    ds = fc.build_dataset(buoys=(fc.TARGET_BUOY,))
    X = fc.build_feature_matrix(ds, value_cols=fc.target_value_cols(ds))

    # pre-registered candidates (recipes frozen by select_backtest.py)
    candidates = [
        ("persistence", None, lambda h, s: fc.Persistence(y, h)),
        ("seasonal_mean", None, lambda h, s: fc.SeasonalMean(y, h)),
        ("ridge_primary", X, lambda h, s: fc.RidgeForecaster(alpha=10.0)),
        ("hgb_primary", X, lambda h, s: fc.HGBForecaster(max_iter=300, learning_rate=0.06,
                                                         min_samples_leaf=200, random_state=s)),
    ]
    for name, Xc, fac in candidates:
        res = fc.evaluate_and_log(fac, y_full=y, X=Xc, splitter=splitter, horizons=HORIZONS_H,
                                  name=name, mode="confirm", n_boot=500,
                                  data_sources=["mooloolaba"])
        _report(res, name)

    if args.nn_arch:
        from forecast import neural
        ds = fc.build_dataset(buoys=(fc.TARGET_BUOY,))
        res = fc.evaluate_and_log(
            lambda h, s: neural.SeqForecaster(ds, PRIMARY_CHANNELS,
                                              context_len=neural.context_for_horizon(h),
                                              horizon_h=h, arch=args.nn_arch, hidden=48,
                                              layers=2, dropout=0.1, epochs=20,
                                              train_stride=3, seed=s),
            y_full=y, X=None, splitter=splitter, horizons=HORIZONS_H,
            name=f"nn_{args.nn_arch}", mode="confirm", seeds=(0, 1), n_boot=500,
            data_sources=["mooloolaba"])
        _report(res, f"nn_{args.nn_arch}")


if __name__ == "__main__":
    main()
