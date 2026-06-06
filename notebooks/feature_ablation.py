"""Per-station feature ablation across horizons × model families.

Run:
    ./.venv/bin/python notebooks/feature_ablation.py --family ridge
    ./.venv/bin/python notebooks/feature_ablation.py --family hgb
    ./.venv/bin/python notebooks/feature_ablation.py --family gru
    ./.venv/bin/python notebooks/feature_ablation.py --family all   # ridge+hgb+gru

Optional restrictors (useful for smoke-testing one cell before the full sweep):
    --horizon 12        Only run h=12
    --skip-ceiling      Skip the all-stations ceiling run
    --skip-baseline     Skip the primary-only baseline run

For each (family, horizon) the sweep runs exactly 24 cells: 1 baseline (primary
buoy only), 1 ceiling (every wave neighbour + every wind station), 11 add-one
runs (primary + one extra station), 11 drop-one runs (everything but one
station). Every run uses the SAME fixed window — see ``forecast.ablation``.

Per-cell `extra` payload includes ``direction`` ("baseline"/"ceiling"/"add"/
"drop"), ``station`` (None for baseline/ceiling), ``family``, and ``horizon_h``
so the report script (``feature_ablation_report.py``) can pivot/aggregate the
JSONL log directly.

Logged with the ``ablate_`` name prefix so it never collides with the existing
``hsweep_`` / ``lineopt_`` / ``seqsweep_`` entries.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import forecast as fc
from forecast import ablation as ab
from forecast.metrics import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

HORIZONS_H: list[int] = [6, 12, 24, 36, 48, 72]
TEST_START = "2023-01-01"
RUN_PREFIX = "ablate"

# Hyperparams locked to the ones used in horizon_sweep.py for direct comparability.
RIDGE_KW = {"alpha": 1.0}
HGB_KW = {
    "max_iter": 800, "learning_rate": 0.03, "max_depth": 6,
    "min_samples_leaf": 50, "l2_regularization": 1.0, "random_state": 42,
    "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 40,
}
# GRU config from notebooks/horizon_sweep.py SEQ_CONFIGS[("baseline","gru")] —
# the best seq-sweep config for the broader-source combo. seq_len=48 = 24h
# of 30-min context; small hidden + 1 layer + 2 epochs is what the sweep
# converged on.
GRU_CFG = {
    "seq_len": 48, "hidden": 64, "num_layers": 1,
    "epochs": 2, "weight_decay": 0.0, "rnn_dropout": 0.0,
    "lr": 1e-3, "batch_size": 512, "seed": 42, "scaler": "robust",
}


def _pinned_split(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    return int(idx.searchsorted(ts))


def _cell_name(family: str, h: int, direction: str, station: str | None) -> str:
    stem = f"{RUN_PREFIX}_{family}_h{h}h_{direction}"
    return stem if station is None else f"{stem}_{station}"


# ---------------------------------------------------------------------------
# Per-family cell runners — each returns the EvaluationResult or skips logging
# when the cell is a no-op.
# ---------------------------------------------------------------------------


def run_ridge_cell(
    sources: ab.PreloadedSources,
    horizon_h: int,
    direction: str,
    stations: list[str],
    station: str | None,
    persist_preds: np.ndarray,
    y_tr: pd.Series,
    y_te: pd.Series,
    extra_base: dict,
) -> dict[str, float]:
    X = ab.build_engineered_design(sources, stations)
    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos = _pinned_split(X.index, ts)
    X_tr, X_te = X.iloc[:pos], X.iloc[pos:]
    preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_tr)
    X_tr_imp, X_te_imp = preproc.transform(X_tr), preproc.transform(X_te)
    sources_list = [ab.PRIMARY_BUOY] + stations
    result = fc.evaluate_and_log(
        Ridge(**RIDGE_KW), X_tr_imp, y_tr, X_te_imp, y_te,
        name=_cell_name("ridge", horizon_h, direction, station),
        baseline_preds=persist_preds,
        data_sources=sources_list,
        extra={**extra_base, "family": "ridge", "direction": direction,
               "station": station, "n_features": int(X_tr_imp.shape[1])},
    )
    return result.metrics


def run_hgb_cell(
    sources: ab.PreloadedSources,
    horizon_h: int,
    direction: str,
    stations: list[str],
    station: str | None,
    persist_preds: np.ndarray,
    y_tr: pd.Series,
    y_te: pd.Series,
    wave_level_tr: pd.Series,
    wave_level_te: pd.Series,
    extra_base: dict,
) -> dict[str, float]:
    """HGB fit on persistence-residual (mirrors horizon_sweep.py)."""
    X = ab.build_engineered_design(sources, stations)
    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos = _pinned_split(X.index, ts)
    X_tr, X_te = X.iloc[:pos], X.iloc[pos:]
    preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_tr)
    X_tr_raw = X_tr[preproc.kept_columns_]
    X_te_raw = X_te[preproc.kept_columns_]
    y_res = y_tr - wave_level_tr
    mask = ~y_res.isna() & ~X_tr_raw.isna().any(axis=1)
    model = HistGradientBoostingRegressor(**HGB_KW)
    model.fit(X_tr_raw.loc[mask].to_numpy(), y_res.loc[mask].to_numpy())
    hgb_preds = wave_level_te.to_numpy() + model.predict(X_te_raw.to_numpy())
    metrics = summarise(y_te.to_numpy(), hgb_preds, y_pred_baseline=persist_preds)
    sources_list = [ab.PRIMARY_BUOY] + stations
    name = _cell_name("hgb", horizon_h, direction, station)
    fc.log_run(
        fc.EvaluationResult(name=name, metrics=metrics, predictions=hgb_preds, model=model),
        data_sources=sources_list,
        train_index=X_tr.index, test_index=X_te.index,
        n_features=int(X_tr_raw.shape[1]),
        extra={**extra_base, "family": "hgb", "direction": direction,
               "station": station, "n_features": int(X_tr_raw.shape[1]),
               "hgb_mode": "persistence_residual"},
    )
    return metrics


def run_gru_cell(
    sources: ab.PreloadedSources,
    horizon_h: int,
    direction: str,
    stations: list[str],
    station: str | None,
    persist_preds: np.ndarray,
    y_tr: pd.Series,
    y_te: pd.Series,
    extra_base: dict,
) -> dict[str, float]:
    X = ab.build_seq_design(sources, stations)
    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos = _pinned_split(X.index, ts)
    X_tr, X_te = X.iloc[:pos], X.iloc[pos:]
    X_tr_imp, X_te_imp = fc.mean_impute(X_tr, X_te)
    device = fc.auto_device()
    model = fc.GRUForecaster(
        seq_len=GRU_CFG["seq_len"], hidden=GRU_CFG["hidden"],
        num_layers=GRU_CFG["num_layers"], epochs=GRU_CFG["epochs"],
        batch_size=GRU_CFG["batch_size"], lr=GRU_CFG["lr"],
        seed=GRU_CFG["seed"], device=device, verbose=False,
        scaler=GRU_CFG["scaler"],
        weight_decay=GRU_CFG["weight_decay"],
        rnn_dropout=GRU_CFG["rnn_dropout"],
    )
    t0 = time.time()
    model.fit(X_tr_imp, y_tr)
    preds = model.predict(X_te_imp)
    elapsed = time.time() - t0
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=persist_preds)
    sources_list = [ab.PRIMARY_BUOY] + stations
    name = _cell_name("gru", horizon_h, direction, station)
    fc.log_run(
        fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=model),
        data_sources=sources_list,
        train_index=X_tr.index, test_index=X_te.index,
        n_features=int(X_tr_imp.shape[1]),
        extra={**extra_base, "family": "gru", "direction": direction,
               "station": station, "n_features": int(X_tr_imp.shape[1]),
               "device": device, "elapsed_min": round(elapsed / 60, 2),
               **{k: v for k, v in GRU_CFG.items() if k != "seed"}},
    )
    return metrics


# ---------------------------------------------------------------------------
# Per-horizon sweep driver
# ---------------------------------------------------------------------------


def run_horizon(
    sources: ab.PreloadedSources,
    horizon_h: int,
    families: list[str],
    *,
    do_baseline: bool = True,
    do_ceiling: bool = True,
) -> None:
    horizon_steps = fc.hours_to_steps(horizon_h)
    y = fc.make_target(sources.wave, horizon_steps=horizon_steps)

    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos = _pinned_split(sources.wave.index, ts)
    y_tr, y_te = y.iloc[:pos], y.iloc[pos:]

    # Persistence on the primary buoy at this horizon — single baseline used
    # for skill scoring across every cell at this horizon.
    X_p = sources.wave[["hsig_m"]]
    X_p_tr, X_p_te = X_p.iloc[:pos], X_p.iloc[pos:]
    persist = fc.evaluate(
        fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te,
        name=f"persistence_h{horizon_h}h",
    )
    pp = persist.predictions

    wave_level_tr = sources.wave["hsig_m"].iloc[:pos]
    wave_level_te = sources.wave["hsig_m"].iloc[pos:]

    extra_base = {
        "horizon_h": horizon_h,
        "horizon_steps": horizon_steps,
        "ablation_window_start": sources.window_start.isoformat(),
        "ablation_window_end": sources.window_end.isoformat(),
        "window": f"{sources.wave.index.min().date()}:{sources.wave.index.max().date()}",
        "imputation": "mean", "scaling": "robust",
    }

    print(f"\n  --- h={horizon_h}h  persistence RMSE {persist.metrics['RMSE']:.4f} ---")

    cells: list[tuple[str, str | None, list[str]]] = []
    if do_baseline:
        cells.append(("baseline", None, []))
    if do_ceiling:
        cells.append(("ceiling", None, ab.ALL_STATIONS))
    for s in ab.ALL_STATIONS:
        cells.append(("add", s, [s]))
    for s in ab.ALL_STATIONS:
        cells.append(("drop", s, [x for x in ab.ALL_STATIONS if x != s]))

    for family in families:
        print(f"\n  === family={family} ===")
        t0 = time.time()
        for direction, station, stations in cells:
            t_cell = time.time()
            if family == "ridge":
                metrics = run_ridge_cell(
                    sources, horizon_h, direction, stations, station, pp,
                    y_tr, y_te, extra_base,
                )
            elif family == "hgb":
                metrics = run_hgb_cell(
                    sources, horizon_h, direction, stations, station, pp,
                    y_tr, y_te, wave_level_tr, wave_level_te, extra_base,
                )
            elif family == "gru":
                metrics = run_gru_cell(
                    sources, horizon_h, direction, stations, station, pp,
                    y_tr, y_te, extra_base,
                )
            else:
                raise ValueError(f"Unknown family: {family}")
            tag = direction if station is None else f"{direction:<5} {station}"
            print(f"    [{time.time()-t_cell:6.1f}s] {tag:<25}  "
                  f"RMSE {metrics['RMSE']:.4f}  Skill {metrics.get('SkillVsBaseline', 0):+.4f}")
        print(f"  family={family} took {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="all",
                        choices=["ridge", "hgb", "gru", "ridge+hgb", "all"])
    parser.add_argument("--horizon", type=int, default=None,
                        help="Restrict to this single horizon (default: all 6)")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-ceiling", action="store_true")
    args = parser.parse_args()

    if args.family == "all":
        families = ["ridge", "hgb", "gru"]
    elif args.family == "ridge+hgb":
        families = ["ridge", "hgb"]
    else:
        families = [args.family]

    horizons = [args.horizon] if args.horizon is not None else HORIZONS_H

    t0 = time.time()
    print("Loading sources …")
    sources = ab.load_all_sources()
    print(f"  fixed window: {sources.window_start} → {sources.window_end}")
    print(f"  wave rows: {len(sources.wave):,}")
    print(f"  ALL_STATIONS ({len(ab.ALL_STATIONS)}): {ab.ALL_STATIONS}")

    for h in horizons:
        run_horizon(sources, h, families,
                    do_baseline=not args.skip_baseline,
                    do_ceiling=not args.skip_ceiling)

    print(f"\nTotal sweep time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
