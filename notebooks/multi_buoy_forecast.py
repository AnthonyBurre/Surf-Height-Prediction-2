"""Multi-buoy +12h hsig_m forecast.

Run:  ./.venv/bin/python notebooks/multi_buoy_forecast.py

Strategy
--------
Caloundra, Brisbane and Gold Coast buoys only cover 2024-2025, which falls
entirely in the test set of the full-history 80/20 split used in forecast_v2.
We therefore scope this experiment to the 2024-2025 overlap window and run an
independent 80/20 chronological split within that window.

This lets the models actually *learn* from neighbour features during training
rather than receiving NaN for every training row.

Experiments (all on the same 2024-2025 split)
----------------------------------------------
1. persistence_2024      – persistence baseline for this window
2. ridge_mool_2024       – Ridge, Mooloolaba features only (same engineering
                           as forecast_v2; benchmark for the reduced window)
3. ridge_multi_2024      – Ridge + neighbour lag/rolling features
4. hgb_multi_2024        – HistGradientBoosting + neighbour features (NaN-native
                           so it gracefully handles any residual gaps)

Full-history Ridge context (from experiments.jsonl, 2015-2025 data):
  RMSE 0.265  |  SkillVsBaseline +11.3 %
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

import forecast as fc
from forecast.metrics import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Feature engineering (Mooloolaba features — mirrors forecast_v2)
# ---------------------------------------------------------------------------

LAG_STEPS    = [1, 2, 3, 6, 12, 24, 48, 96, 144]
ROLL_WINDOWS = [12, 24, 48, 96]
DELTA_STEPS  = [6, 12, 24, 48]
NEIGHBOUR_LAGS = [1, 2, 3, 6, 12, 24]  # 30-min steps; raw col = lag 0
NEIGHBOUR_ROLL = [6, 12, 24]            # rolling windows for neighbours


def add_momentum(df: pd.DataFrame, columns: list[str], deltas: list[int]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for d in deltas:
            out[f"{col}_delta_{d}"] = df[col] - df[col].shift(d)
    return out


def mool_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full Mooloolaba feature set (identical to forecast_v2.build_features)."""
    return (
        df.pipe(fc.encode_circular)
          .pipe(fc.add_time_features)
          .pipe(fc.add_lag_features,
                columns=["hsig_m", "hmax_m", "tp_s", "tz_s"], lags=LAG_STEPS)
          .pipe(fc.add_rolling_features,
                columns=["hsig_m", "tp_s", "hmax_m"],
                windows=ROLL_WINDOWS, stats=("mean", "std", "min", "max"))
          .pipe(add_momentum,
                columns=["hsig_m", "tp_s", "hmax_m"], deltas=DELTA_STEPS)
    )


def neighbour_features(
    X_mool: pd.DataFrame, merged: pd.DataFrame, neighbour_cols: list[str]
) -> pd.DataFrame:
    """Append raw + lag + rolling features for each neighbour hsig_m column.

    Reads neighbour raw values from merged (which has them) and appends onto
    X_mool (which does not). This keeps the Mooloolaba-only and multi-buoy
    feature matrices cleanly separated.
    """
    out = X_mool.copy()
    for col in neighbour_cols:
        out[col] = merged[col]                          # raw value at t (lag 0)
        for lag in NEIGHBOUR_LAGS:
            out[f"{col}_lag{lag}"] = merged[col].shift(lag)
        for w in NEIGHBOUR_ROLL:
            r = merged[col].shift(1).rolling(window=w, min_periods=max(1, w // 2))
            out[f"{col}_roll{w}_mean"] = r.mean()
            out[f"{col}_roll{w}_std"]  = r.std()
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_buoy(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / filename,
        parse_dates=["datetime_utc"],
        index_col="datetime_utc",
    )


def build_merged() -> tuple[pd.DataFrame, list[str]]:
    """Load all four buoys, restrict to overlap, merge into one frame.

    Returns (merged_df, neighbour_hsig_cols).
    Neighbour columns are renamed to '<buoy>_hsig_m' to avoid collision.
    """
    mool = load_buoy("mooloolaba_wave_data_2015-2025.csv")
    neighbours = {
        "caloundra": load_buoy("caloundra_wave_data_2024-2025.csv"),
        "brisbane":  load_buoy("brisbane_wave_data_2024-2025.csv"),
        "goldcoast": load_buoy("gold-coast_wave_data_2024-2025.csv"),
    }

    # Restrict Mooloolaba to the overlap window shared by all neighbours.
    start = max(df.index.min() for df in neighbours.values())
    end   = min(df.index.max() for df in neighbours.values())
    mool  = mool.loc[start:end]
    print(f"Overlap window : {start}  →  {end}")
    print(f"Mooloolaba rows: {len(mool):,}")

    # Build merged frame: Mooloolaba base + neighbour hsig_m columns.
    merged = mool.copy()
    neighbour_cols: list[str] = []
    for name, nb in neighbours.items():
        nb = nb.loc[start:end]
        col = f"{name}_hsig_m"
        merged[col] = nb["hsig_m"]
        neighbour_cols.append(col)
        nan_pct = merged[col].isna().mean() * 100
        print(f"  {name:10s}: {len(nb):,} rows  |  NaN in merged: {nan_pct:.1f}%")

    return merged, neighbour_cols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(name, preds, y_te, baseline_preds, model=None):
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=baseline_preds)
    return fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=model)


def fit_hgb_direct(kw, X_tr, y_tr, X_te):
    mask = ~y_tr.isna()
    m = HistGradientBoostingRegressor(**kw)
    m.fit(X_tr.loc[mask].to_numpy(), y_tr.loc[mask].to_numpy())
    return m.predict(X_te.to_numpy()), m


def mean_impute(
    X_tr: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a mean imputer on X_tr, transform both. Returns new DataFrames."""
    imp = SimpleImputer(strategy="mean")
    X_tr_imp = pd.DataFrame(
        imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
    )
    X_te_imp = pd.DataFrame(
        imp.transform(X_te), columns=X_te.columns, index=X_te.index
    )
    return X_tr_imp, X_te_imp


def timed(label, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    print(f"  [{time.time() - t0:5.1f}s] {label}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    merged, neighbour_cols = build_merged()

    y = fc.make_target(merged)

    # --- Feature matrices ---------------------------------------------------
    # Slice to Mooloolaba-only columns before building X_mool so that neighbour
    # raw columns are not silently included in the "mool-only" baseline.
    mool_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
    X_mool  = mool_features(mool_only)
    X_multi = neighbour_features(X_mool, merged, neighbour_cols)

    # Persistence only needs hsig_m; split it for the baseline.
    X_p = merged[["hsig_m"]]

    # All splits use the same 80/20 chronological boundary.
    X_mool_tr,  X_mool_te,  y_tr, y_te = fc.chronological_split(X_mool, y)
    X_multi_tr, X_multi_te, _,    _    = fc.chronological_split(X_multi, y)
    X_p_tr,     X_p_te,     _,    _    = fc.chronological_split(X_p, y)

    print(f"\nRows : {len(merged):,}  |  train: {len(X_mool_tr):,}  test: {len(X_mool_te):,}")
    print(f"Mooloolaba features : {X_mool.shape[1]}")
    print(f"Multi-buoy features : {X_multi.shape[1]}  "
          f"(+{X_multi.shape[1] - X_mool.shape[1]} neighbour features)\n")

    results: list[fc.EvaluationResult] = []

    # -----------------------------------------------------------------------
    print("=== Baselines ===")
    persist = timed("persistence",
        fc.evaluate_and_log, fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name="persistence_2024",
        data_sources=["mooloolaba"],
        extra={"window": "2024-2025"})
    results.append(persist)
    pp = persist.predictions

    # Pre-impute (mean, fit on train) so Ridge's all-or-nothing NaN mask doesn't
    # kill every test row — sst_c drops out partway through 2025.
    X_mool_tr_imp,  X_mool_te_imp  = mean_impute(X_mool_tr,  X_mool_te)
    X_multi_tr_imp, X_multi_te_imp = mean_impute(X_multi_tr, X_multi_te)

    # -----------------------------------------------------------------------
    print("\n=== Mooloolaba-only Ridge (2024-2025 window) ===")
    ridge_mool = timed("ridge_mool_2024",
        fc.evaluate_and_log, Ridge(alpha=1.0),
        X_mool_tr_imp, y_tr, X_mool_te_imp, y_te,
        name="ridge_mool_2024",
        baseline_preds=pp,
        data_sources=["mooloolaba"],
        extra={"window": "2024-2025",
               "note": "same features as forecast_v2, reduced time window",
               "imputation": "mean"})
    results.append(ridge_mool)

    # -----------------------------------------------------------------------
    print("\n=== Multi-buoy Ridge ===")
    ridge_multi = timed("ridge_multi_2024",
        fc.evaluate_and_log, Ridge(alpha=1.0),
        X_multi_tr_imp, y_tr, X_multi_te_imp, y_te,
        name="ridge_multi_2024",
        baseline_preds=pp,
        data_sources=["mooloolaba", "caloundra", "brisbane", "goldcoast"],
        extra={"window": "2024-2025",
               "neighbour_features": neighbour_cols,
               "imputation": "mean"})
    results.append(ridge_multi)

    # -----------------------------------------------------------------------
    print("\n=== Multi-buoy HGB (NaN-native) ===")
    hgb_kw = dict(
        max_iter=800, learning_rate=0.03, max_depth=6,
        min_samples_leaf=50, l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=40,
    )
    t0 = time.time()
    hgb_preds, hgb_model = fit_hgb_direct(hgb_kw, X_multi_tr, y_tr, X_multi_te)
    print(f"  [{time.time() - t0:5.1f}s] hgb_multi_2024")
    hgb_result = make_result("hgb_multi_2024", hgb_preds, y_te, pp, model=hgb_model)
    fc.log_run(hgb_result,
               data_sources=["mooloolaba", "caloundra", "brisbane", "goldcoast"],
               train_index=X_multi_tr.index, test_index=X_multi_te.index,
               n_features=X_multi_tr.shape[1],
               extra={"window": "2024-2025", "nan_handling": "native_hgb",
                      "neighbour_features": neighbour_cols})
    results.append(hgb_result)

    # -----------------------------------------------------------------------
    print("\n=== Feature importance (HGB, top 20) ===")
    if hasattr(hgb_model, "feature_importances_"):
        feat_imp = pd.Series(
            hgb_model.feature_importances_,
            index=X_multi_tr.columns,
        ).sort_values(ascending=False)
        print(feat_imp.head(20).to_string())
    else:
        print("  (feature_importances_ not available for this sklearn version)")

    # -----------------------------------------------------------------------
    print("\n=== Results (sorted by RMSE) ===")
    table = fc.compare(results).round(4)
    print(table.to_string())

    # Context: best full-history result from experiments.jsonl.
    # The 'metrics' column is a dict; access it directly.
    log = fc.read_log()
    full_hist_row = log[log["name"] == "ridge"].sort_values("timestamp").iloc[-1]
    m = full_hist_row["metrics"]
    print(f"\nContext — full-history Ridge (2015-2025):")
    print(f"  RMSE {m['RMSE']:.4f}  Skill {m['SkillVsBaseline']:+.4f}")

    return table


if __name__ == "__main__":
    main()
