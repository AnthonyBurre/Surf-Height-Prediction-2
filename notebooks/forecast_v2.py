"""+12h hsig_m forecast — model comparison v2.

Run:  ./.venv/bin/python notebooks/forecast_v2.py

Headline finding: persistence is a stiff baseline at this horizon (12h
autocorrelation of hsig_m ≈ 0.81). The best model here is **Ridge with rich
engineered features** at ~+11% skill score over persistence. Tree-based and
sequence models did not match it on this dataset:

  - HistGradientBoosting (NaN-tolerant, direct fit) underperforms Ridge by
    several percent. The signal here is dominated by linear combinations of
    lag/rolling features; the non-linear capacity of trees mostly captures
    noise.
  - LSTM / GRU on raw circular-encoded channels regress hard toward the mean
    (predicted std ~80% of target std). Without the engineered lag/momentum
    features, the seq-model has to rediscover temporal structure from 7
    inputs * 48 timesteps, and ~150k windows isn't enough.

Where additional skill probably lives (out of scope here):
  - More data sources: BOM/GFS atmospheric reanalysis, neighbouring buoys.
    Atmospheric forcing leads buoy response; a wind/pressure feature would
    plausibly add several percent.
  - Probabilistic forecasts (quantile regression / conformal intervals).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import forecast as fc
from forecast.metrics import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
# Ensemble averages over rows where both members are NaN — np.nanmean
# correctly returns NaN there but spams a RuntimeWarning. Silence it.
warnings.filterwarnings("ignore", message="Mean of empty slice")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

LAG_STEPS = [1, 2, 3, 6, 12, 24, 48, 96, 144]  # 30min … 72h
ROLL_WINDOWS = [12, 24, 48, 96]                 # 6h, 12h, 24h, 48h
DELTA_STEPS = [6, 12, 24, 48]                   # 3h, 6h, 12h, 24h


def add_momentum(df: pd.DataFrame, columns: list[str], deltas: list[int]) -> pd.DataFrame:
    """col(t) - col(t - d) for every (col, d). Captures trend / direction."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(name: str, preds: np.ndarray, y_te: pd.Series,
                baseline_preds: np.ndarray) -> fc.EvaluationResult:
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=baseline_preds)
    return fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=None)


def fit_hgb_direct(kw: dict, X_tr: pd.DataFrame, y_tr: pd.Series,
                   X_te: pd.DataFrame) -> np.ndarray:
    """HGB tolerates NaN in X natively, so skip the harness mask and recover
    the rows it would otherwise drop (here ~12% — driven by SST's 10% NaN
    rate)."""
    mask = ~y_tr.isna()
    m = HistGradientBoostingRegressor(**kw)
    m.fit(X_tr.loc[mask].to_numpy(), y_tr.loc[mask].to_numpy())
    return m.predict(X_te.to_numpy())


def timed(label: str, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    print(f"  [{time.time() - t0:5.1f}s] {label}")
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    df = fc.load_data()
    y = fc.make_target(df)
    X_eng = build_features(df)

    X_eng_tr, X_eng_te, y_tr, y_te = fc.chronological_split(X_eng, y)
    X_p_tr, X_p_te, _, _ = fc.chronological_split(df[["hsig_m"]], y)

    print(f"rows={len(df):,}  features={X_eng.shape[1]}  "
          f"train={len(X_eng_tr):,}  test={len(X_eng_te):,}\n")

    results: list[fc.EvaluationResult] = []

    # Persistence — skill reference for everything below
    print("Baselines:")
    persist = timed("persistence",
        fc.evaluate, fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te, name="persistence")
    results.append(persist)
    pp = persist.predictions

    # Diagnostic floors (expected to be much worse than persistence)
    results.append(timed("seasonal_naive_24h",
        fc.evaluate, fc.SeasonalNaiveForecaster(period_steps=48),
        X_p_tr, y_tr, X_p_te, y_te,
        name="seasonal_naive_24h", baseline_preds=pp))
    results.append(timed("climatology_hour",
        fc.evaluate, fc.ClimatologyHourForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name="climatology_hour", baseline_preds=pp))

    # Ridge — the winner. With 107 engineered features and ~150k rows the
    # problem is well-conditioned; alpha barely matters (sweep was flat).
    print("\nRidge:")
    ridge_result = timed("ridge",
        fc.evaluate, Ridge(alpha=1.0),
        X_eng_tr, y_tr, X_eng_te, y_te,
        name="ridge", baseline_preds=pp)
    results.append(ridge_result)

    # HGB direct (NaN-tolerant) — runs on the same feature set, recovers the
    # SST-NaN rows the harness drops. Still doesn't beat Ridge on this data.
    print("\nHGB (direct, NaN-tolerant):")
    hgb_kw = dict(
        max_iter=800, learning_rate=0.03, max_depth=6,
        min_samples_leaf=200, l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=40,
    )
    t0 = time.time()
    hgb_preds = fit_hgb_direct(hgb_kw, X_eng_tr, y_tr, X_eng_te)
    print(f"  [{time.time() - t0:5.1f}s] hgb_direct")
    results.append(make_result("hgb_direct", hgb_preds, y_te, pp))

    # HGB on persistence residuals — predict (y - hsig_m_now), add hsig_m back.
    # Forces the model to learn only the correction over persistence.
    print("\nHGB on persistence residuals:")
    hsig_now_tr = df["hsig_m"].loc[X_eng_tr.index].to_numpy()
    hsig_now_te = df["hsig_m"].loc[X_eng_te.index].to_numpy()
    y_resid = y_tr.to_numpy() - hsig_now_tr
    mask = np.isfinite(y_resid)

    t0 = time.time()
    hgb_resid = HistGradientBoostingRegressor(**hgb_kw)
    hgb_resid.fit(X_eng_tr.loc[mask].to_numpy(), y_resid[mask])
    delta = hgb_resid.predict(X_eng_te.to_numpy())
    hgb_persist_resid = hsig_now_te + delta
    print(f"  [{time.time() - t0:5.1f}s] hgb_persistence_residual")
    results.append(make_result("hgb_persistence_residual", hgb_persist_resid, y_te, pp))

    # Ensemble of the two non-baseline survivors.
    ridge_p = ridge_result.predictions
    stack = np.vstack([ridge_p, hgb_persist_resid])
    ens = np.nanmean(stack, axis=0)
    results.append(make_result("ensemble_ridge_hgbresid", ens, y_te, pp))

    print("\n=== Results (sorted by RMSE) ===")
    table = fc.compare(results).round(4)
    print(table.to_string())
    return table


if __name__ == "__main__":
    main()
