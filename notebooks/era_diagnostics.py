"""Investigation: why does the 2024-2025 window produce better forecasts?

Run:  ./.venv/bin/python notebooks/era_diagnostics.py

Three hypotheses
----------------
A. Sea-state difficulty — 2024-2025 is intrinsically calmer / more predictable
   (lower variance, higher autocorrelation). Every model benefits.
B. Distribution shift — full model trains on 2015-2023 but tests on 2023-2025.
   If the wave regime drifted, it is predicting partly out-of-distribution.
C. Recency — training close in time to the test period helps regardless
   of regime (the model learns the current seasonal pattern better).

Steps
-----
1. Persistence RMSE and hsig_m statistics year-by-year (screen A).
2. Autocorrelation at the forecast horizon by year (also screens A).
3. Cross-evaluation (screens B and C):
   - Full-history Ridge  → evaluate on the 2024-2025 test window.
   - 2024-2025-scoped Ridge → evaluate on the full-history test window.
   Both models evaluated on the SHARED window (Jul-Nov 2025) for a clean
   apples-to-apples read.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))  # expose notebooks/ for imports

import forecast as fc
from forecast.metrics import mae, rmse, summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

DATA_DIR = Path(__file__).parent.parent / "data"

# Feature engineering — mirrors forecast_v2 exactly.
LAG_STEPS    = [1, 2, 3, 6, 12, 24, 48, 96, 144]
ROLL_WINDOWS = [12, 24, 48, 96]
DELTA_STEPS  = [6, 12, 24, 48]


def add_momentum(df, columns, deltas):
    out = df.copy()
    for col in columns:
        for d in deltas:
            out[f"{col}_delta_{d}"] = df[col] - df[col].shift(d)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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


def mean_impute(
    X_tr: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = SimpleImputer(strategy="mean")
    X_tr_i = pd.DataFrame(
        imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
    )
    X_te_i = pd.DataFrame(
        imp.transform(X_te), columns=X_te.columns, index=X_te.index
    )
    return X_tr_i, X_te_i, imp


def persistence_rmse_on(hsig: pd.Series) -> float:
    """Persistence RMSE for a hsig_m series (drops NaN pairs automatically)."""
    pred = hsig.shift(fc.HORIZON_STEPS)  # predict t using value at t-24
    actual = hsig
    mask = ~(actual.isna() | pred.isna())
    return float(np.sqrt(np.mean((actual[mask].values - pred[mask].values) ** 2)))


# ---------------------------------------------------------------------------
# Step 1 + 2 — Year-by-year sea-state statistics
# ---------------------------------------------------------------------------

def step1_year_stats(mool: pd.DataFrame) -> pd.DataFrame:
    """Compute per-year persistence RMSE, hsig_m stats, and autocorrelation."""
    print("=" * 60)
    print("STEP 1+2 — Year-by-year sea-state statistics")
    print("=" * 60)

    rows = []
    for year in range(2015, 2026):
        s = mool.loc[str(year), "hsig_m"].dropna()
        if len(s) < 100:
            continue

        # Persistence RMSE restricted to this year's rows
        p_rmse = persistence_rmse_on(mool.loc[str(year), "hsig_m"])

        # Autocorrelation at the +12h horizon (24 steps)
        # acf at lag k = corr(s[t], s[t+k])
        acf_12h = s.autocorr(lag=fc.HORIZON_STEPS)

        rows.append({
            "year":         year,
            "n_obs":        len(s),
            "mean_m":       s.mean(),
            "std_m":        s.std(),
            "p90_m":        s.quantile(0.90),
            "acf_12h":      acf_12h,
            "persist_rmse": p_rmse,
        })

    df = pd.DataFrame(rows).set_index("year")

    print(df.round(3).to_string())

    # Summary: is 2024-2025 materially different?
    old = df.loc[2015:2022]
    new = df.loc[2024:2025]
    print(f"\nPre-2024 avg persistence RMSE : {old['persist_rmse'].mean():.4f}")
    print(f"2024-2025 avg persistence RMSE: {new['persist_rmse'].mean():.4f}")
    print(f"Pre-2024 avg std(hsig_m)       : {old['std_m'].mean():.4f}")
    print(f"2024-2025 avg std(hsig_m)       : {new['std_m'].mean():.4f}")
    print(f"Pre-2024 avg acf_12h           : {old['acf_12h'].mean():.4f}")
    print(f"2024-2025 avg acf_12h           : {new['acf_12h'].mean():.4f}")

    return df


# ---------------------------------------------------------------------------
# Step 3 — Cross-evaluation
# ---------------------------------------------------------------------------

def train_ridge(X_tr: pd.DataFrame, y_tr: pd.Series) -> tuple[Ridge, SimpleImputer]:
    """Fit Ridge(alpha=1) with mean imputation; return (model, imputer)."""
    # Drop rows where the TARGET is NaN before fitting.
    valid = ~y_tr.isna()
    imp = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(
        imp.fit_transform(X_tr[valid]), columns=X_tr.columns, index=X_tr[valid].index
    )
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_imp, y_tr[valid])
    return ridge, imp


def score_on(
    model: Ridge,
    imp: SimpleImputer,
    X: pd.DataFrame,
    y: pd.Series,
    persist_rmse: float,
    label: str,
) -> dict:
    """Apply imputer → predict → score on the given window."""
    X_imp = pd.DataFrame(imp.transform(X), columns=X.columns, index=X.index)
    preds = model.predict(X_imp)
    mask = ~y.isna()
    m = summarise(y[mask].values, preds[mask], y_pred_baseline=None)
    skill = 1.0 - (m["RMSE"] ** 2) / (persist_rmse ** 2)
    print(f"  {label:<45s}  RMSE={m['RMSE']:.4f}  "
          f"MAE={m['MAE']:.4f}  skill={skill:+.4f}")
    return {"label": label, **m, "skill_vs_persist": skill}


def step3_cross_eval(mool: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STEP 3 — Cross-evaluation")
    print("=" * 60)

    y_full = fc.make_target(mool)
    X_full = build_features(mool)

    # --- Full-history split (mirrors forecast_v2) ---------------------------
    X_fh_tr, X_fh_te, y_fh_tr, y_fh_te = fc.chronological_split(X_full, y_full)
    print(f"\nFull-history  train: {X_fh_tr.index.min().date()} → {X_fh_tr.index.max().date()}"
          f"  ({len(X_fh_tr):,} rows)")
    print(f"Full-history  test : {X_fh_te.index.min().date()} → {X_fh_te.index.max().date()}"
          f"  ({len(X_fh_te):,} rows)")

    # --- 2024-2025-scoped split (mirrors multi_buoy_forecast) ---------------
    cal  = pd.read_csv(DATA_DIR / "caloundra_wave_data_2024-2025.csv",
                       parse_dates=["datetime_utc"], index_col="datetime_utc")
    bris = pd.read_csv(DATA_DIR / "brisbane_wave_data_2024-2025.csv",
                       parse_dates=["datetime_utc"], index_col="datetime_utc")
    gc   = pd.read_csv(DATA_DIR / "gold-coast_wave_data_2024-2025.csv",
                       parse_dates=["datetime_utc"], index_col="datetime_utc")
    ov_start = max(cal.index.min(), bris.index.min(), gc.index.min())
    ov_end   = min(cal.index.max(), bris.index.max(), gc.index.max())

    mool_ov = mool.loc[ov_start:ov_end]
    y_ov    = fc.make_target(mool_ov)
    X_ov    = build_features(mool_ov)
    X_sc_tr, X_sc_te, y_sc_tr, y_sc_te = fc.chronological_split(X_ov, y_ov)
    print(f"\n2024-2025     train: {X_sc_tr.index.min().date()} → {X_sc_tr.index.max().date()}"
          f"  ({len(X_sc_tr):,} rows)")
    print(f"2024-2025     test : {X_sc_te.index.min().date()} → {X_sc_te.index.max().date()}"
          f"  ({len(X_sc_te):,} rows)")

    # Shared evaluation window = the 2024-2025 test window (sits inside both).
    shared_start = X_sc_te.index.min()
    shared_end   = X_sc_te.index.max()
    print(f"\nShared eval window: {shared_start.date()} → {shared_end.date()}")

    # --- Train both models --------------------------------------------------
    print("\nTraining full-history Ridge …", end=" ", flush=True)
    ridge_fh, imp_fh = train_ridge(X_fh_tr, y_fh_tr)
    print("done.")

    print("Training 2024-2025-scoped Ridge …", end=" ", flush=True)
    ridge_sc, imp_sc = train_ridge(X_sc_tr, y_sc_tr)
    print("done.")

    # --- Compute persistence RMSE for each evaluation window ----------------
    persist_fh     = persistence_rmse_on(mool["hsig_m"].loc[X_fh_te.index])
    persist_sc     = persistence_rmse_on(mool["hsig_m"].loc[X_sc_te.index])
    persist_shared = persistence_rmse_on(mool["hsig_m"].loc[shared_start:shared_end])

    print(f"\nPersistence RMSE:")
    print(f"  Full-history test window  : {persist_fh:.4f}")
    print(f"  2024-2025 test window     : {persist_sc:.4f}")
    print(f"  Shared window             : {persist_shared:.4f}")

    # Align features to the shared window using the FULL feature frame
    # (both models were built on the same raw columns, so this is safe).
    X_shared_fh = X_full.loc[shared_start:shared_end]
    X_shared_sc = X_ov.loc[shared_start:shared_end]
    y_shared    = y_full.loc[shared_start:shared_end]

    print("\n--- Each model on its own test set (sanity-check) ---")
    score_on(ridge_fh, imp_fh, X_fh_te, y_fh_te,    persist_fh,
             "full-history Ridge → full-history test")
    score_on(ridge_sc, imp_sc, X_sc_te, y_sc_te,    persist_sc,
             "scoped Ridge → 2024-2025 test")

    print("\n--- Both models on the SHARED window (Jul-Nov 2025) ---")
    r_fh_shared = score_on(ridge_fh, imp_fh, X_shared_fh, y_shared, persist_shared,
                            "full-history Ridge → shared window")
    r_sc_shared = score_on(ridge_sc, imp_sc, X_shared_sc, y_shared, persist_shared,
                            "scoped Ridge → shared window")

    rmse_gap = r_fh_shared["RMSE"] - r_sc_shared["RMSE"]
    print(f"\n  RMSE gap (full-hist minus scoped) on shared window: {rmse_gap:+.4f}")
    if abs(rmse_gap) < 0.005:
        verdict = "A (sea-state difficulty — both models similar on shared window)"
    elif rmse_gap > 0:
        verdict = "B/C (distribution shift or recency — full-history model worse on recent data)"
    else:
        verdict = "Unexpected: full-history model BETTER on recent data"
    print(f"  → Primary hypothesis supported: {verdict}")

    print("\n--- Scoped Ridge on the full-history test window ---")
    # Restrict the full-history test set to dates the scoped model's features cover.
    X_fh_te_ov = X_full.loc[X_fh_te.index.intersection(X_ov.index)]
    y_fh_te_ov = y_full.loc[X_fh_te_ov.index]
    X_ov_aligned = X_ov.loc[X_fh_te_ov.index]
    persist_fh_ov = persistence_rmse_on(mool["hsig_m"].loc[X_fh_te_ov.index])
    score_on(ridge_sc, imp_sc, X_ov_aligned, y_fh_te_ov, persist_fh_ov,
             "scoped Ridge → full-history test (overlap portion)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mool = fc.load_data()
    step1_year_stats(mool)
    step3_cross_eval(mool)


if __name__ == "__main__":
    main()
