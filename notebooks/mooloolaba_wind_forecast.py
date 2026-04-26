"""Mooloolaba +12h hsig_m forecast — Mountain Creek wind addition.

Run:  ./.venv/bin/python notebooks/mooloolaba_wind_forecast.py

Strategy
--------
Mountain Creek (Sunshine Coast Council AWS at -26.69, 153.10 — effectively
co-located with the Mooloolaba wave buoy) carries hourly 10 m wind speed and
direction back to 2015. The wave history is sliced to 2015-2024 to match the
wind window — a separate persistence baseline is computed on that same window
so skill scores are directly comparable to the wind-augmented runs (the
existing 2015-2025 persistence row in experiments.jsonl is on a different
test split).

The wind frame is reindexed onto the 30-min wave grid by forward-fill: each
30-min slot inherits the most recent past hourly reading (e.g. the 14:30
slot gets the 14:00 wind value), which is strictly past-only.

Wind direction is circular (359° and 1° are 2° apart, not 358°), so it is
sin/cos-encoded before being passed to add_neighbour_features — same pattern
as the wave-buoy peak_dir_deg encoding in build_mooloolaba_features.

Models (all on the 2015-2024 window, 80/20 chronological split)
---------------------------------------------------------------
1. persistence_2015_2024     — baseline scoped to this window
2. ridge_mool_2015_2024      — Ridge, Mooloolaba features only
3. ridge_wind_2015_2024      — Ridge + Mountain Creek wind features
4. lasso_wind_2015_2024      — Lasso (sparse linear) + wind, for variable
                               selection / interpretability
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge

import forecast as fc
from forecast.features import encode_circular

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

DATA_DIR = Path(__file__).parent.parent / "data"


def load_wave_2015_2024() -> pd.DataFrame:
    df = fc.load_data()
    return df.loc[df.index.year <= 2024]


def load_wind_aligned(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly Mountain Creek wind, sin/cos-encoded for direction, reindexed
    onto the 30-min wave grid by past-only forward-fill."""
    wind = pd.read_csv(
        DATA_DIR / "mountain-creek_wind_data_2015-2024.csv",
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    )
    wind = encode_circular(wind, columns=["wind_dir_deg"])
    return wind.reindex(target_index, method="ffill")


def mean_impute(
    X_tr: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = SimpleImputer(strategy="mean")
    return (
        pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index),
        pd.DataFrame(imp.transform(X_te),     columns=X_te.columns, index=X_te.index),
    )


def timed(label, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    print(f"  [{time.time() - t0:5.1f}s] {label}")
    return out


def main() -> pd.DataFrame:
    wave = load_wave_2015_2024()
    wind = load_wind_aligned(wave.index)

    # Trim both to their overlap to keep the comparison clean.
    start = max(wave.index.min(), wind.dropna(how="all").index.min())
    end   = min(wave.index.max(), wind.dropna(how="all").index.max())
    wave  = wave.loc[start:end]
    wind  = wind.loc[start:end]

    print(f"Window : {start} → {end}")
    print(f"Wave rows : {len(wave):,}")
    print(f"Wind cols : {list(wind.columns)}")
    nan_pct = wind.isna().mean().mul(100).round(2).to_dict()
    print(f"Wind NaN%: {nan_pct}\n")

    y      = fc.make_target(wave)
    X_mool = fc.build_mooloolaba_features(wave)

    wind_cols = [
        "wind_speed_ms",
        "wind_dir_deg_sin",
        "wind_dir_deg_cos",
        "wind_sigma_theta_deg",
        "wind_speed_std_ms",
    ]
    X_wind = fc.add_neighbour_features(X_mool, wind, wind_cols)
    X_p    = wave[["hsig_m"]]

    X_mool_tr, X_mool_te, y_tr, y_te = fc.chronological_split(X_mool, y)
    X_wind_tr, X_wind_te, _,    _    = fc.chronological_split(X_wind, y)
    X_p_tr,    X_p_te,    _,    _    = fc.chronological_split(X_p,    y)

    print(f"train: {len(X_mool_tr):,}  test: {len(X_mool_te):,}")
    print(f"Mooloolaba features : {X_mool.shape[1]}")
    print(f"+ wind features      : {X_wind.shape[1]}  "
          f"(+{X_wind.shape[1] - X_mool.shape[1]} wind-derived)\n")

    X_mool_tr_imp, X_mool_te_imp = mean_impute(X_mool_tr, X_mool_te)
    X_wind_tr_imp, X_wind_te_imp = mean_impute(X_wind_tr, X_wind_te)

    results: list[fc.EvaluationResult] = []
    window_str = f"{start.date()}:{end.date()}"

    print("=== Persistence baseline (2015-2024) ===")
    persist = timed(
        "persistence_2015_2024",
        fc.evaluate_and_log, fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name="persistence_2015_2024",
        data_sources=["mooloolaba"],
        extra={"window": window_str},
    )
    results.append(persist)
    pp = persist.predictions

    print("\n=== Ridge — Mooloolaba only (2015-2024) ===")
    results.append(timed(
        "ridge_mool_2015_2024",
        fc.evaluate_and_log, Ridge(alpha=1.0),
        X_mool_tr_imp, y_tr, X_mool_te_imp, y_te,
        name="ridge_mool_2015_2024",
        baseline_preds=pp,
        data_sources=["mooloolaba"],
        extra={"window": window_str, "imputation": "mean"},
    ))

    print("\n=== Ridge + wind ===")
    ridge_wind = timed(
        "ridge_wind_2015_2024",
        fc.evaluate_and_log, Ridge(alpha=1.0),
        X_wind_tr_imp, y_tr, X_wind_te_imp, y_te,
        name="ridge_wind_2015_2024",
        baseline_preds=pp,
        data_sources=["mooloolaba", "mountain-creek"],
        extra={"window": window_str, "imputation": "mean",
               "wind_columns": wind_cols},
    )
    results.append(ridge_wind)

    print("\n=== Lasso + wind (sparse linear) ===")
    lasso = timed(
        "lasso_wind_2015_2024",
        fc.evaluate_and_log, Lasso(alpha=0.001, max_iter=10000),
        X_wind_tr_imp, y_tr, X_wind_te_imp, y_te,
        name="lasso_wind_2015_2024",
        baseline_preds=pp,
        data_sources=["mooloolaba", "mountain-creek"],
        extra={"window": window_str, "imputation": "mean", "alpha": 0.001,
               "wind_columns": wind_cols},
    )
    results.append(lasso)

    lasso_coef = lasso.model.coef_
    n_nonzero  = int((lasso_coef != 0).sum())
    print(f"  Non-zero coefficients: {n_nonzero} / {len(lasso_coef)}")
    top = (
        pd.Series(np.abs(lasso_coef), index=X_wind_tr_imp.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    wind_top = [f for f in top.index if "wind" in f]
    print(f"  Wind features in top 15 |coef|: {len(wind_top)}")
    print("  Top 15 by |coef|:")
    for feat, val in top.items():
        print(f"    {feat:42s}  {val:.4f}")

    print("\n=== Results ===")
    table = fc.compare(results).round(4)
    print(table.to_string())

    log = fc.read_log()
    prev_names = ["ridge", "ridge_mool_brisbane", "lasso_mool_brisbane"]
    prev = log[log["name"].isin(prev_names)].copy()
    if not prev.empty:
        print("\nContext from experiments.jsonl (different windows / data sets):")
        for _, row in prev.sort_values("timestamp").drop_duplicates("name", keep="last").iterrows():
            m = row["metrics"]
            print(f"  {row['name']:25s}  RMSE {m['RMSE']:.4f}  Skill {m['SkillVsBaseline']:+.4f}")

    return table


if __name__ == "__main__":
    main()
