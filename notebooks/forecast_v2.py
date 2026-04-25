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
warnings.filterwarnings("ignore", message="Mean of empty slice")


def make_result(name: str, preds: np.ndarray, y_te: pd.Series,
                baseline_preds: np.ndarray,
                model: object | None = None) -> fc.EvaluationResult:
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=baseline_preds)
    return fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=model)


def fit_hgb_direct(kw: dict, X_tr: pd.DataFrame, y_tr: pd.Series,
                   X_te: pd.DataFrame) -> tuple[np.ndarray, HistGradientBoostingRegressor]:
    """HGB tolerates NaN in X natively, so skip the harness mask and recover
    the rows it would otherwise drop (here ~12% — driven by SST's 10% NaN
    rate)."""
    mask = ~y_tr.isna()
    m = HistGradientBoostingRegressor(**kw)
    m.fit(X_tr.loc[mask].to_numpy(), y_tr.loc[mask].to_numpy())
    return m.predict(X_te.to_numpy()), m


def timed(label: str, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    print(f"  [{time.time() - t0:5.1f}s] {label}")
    return out


def main() -> pd.DataFrame:
    df = fc.load_data()
    y = fc.make_target(df)
    X_eng = fc.build_mooloolaba_features(df)

    X_eng_tr, X_eng_te, y_tr, y_te = fc.chronological_split(X_eng, y)
    X_p_tr, X_p_te, _, _ = fc.chronological_split(df[["hsig_m"]], y)

    print(f"rows={len(df):,}  features={X_eng.shape[1]}  "
          f"train={len(X_eng_tr):,}  test={len(X_eng_te):,}\n")

    results: list[fc.EvaluationResult] = []
    data_sources = ["mooloolaba"]

    print("Baselines:")
    persist = timed("persistence",
        fc.evaluate_and_log, fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name="persistence", data_sources=data_sources)
    results.append(persist)
    pp = persist.predictions

    results.append(timed("seasonal_naive_24h",
        fc.evaluate_and_log, fc.SeasonalNaiveForecaster(period_steps=48),
        X_p_tr, y_tr, X_p_te, y_te,
        name="seasonal_naive_24h", baseline_preds=pp, data_sources=data_sources))
    results.append(timed("climatology_hour",
        fc.evaluate_and_log, fc.ClimatologyHourForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name="climatology_hour", baseline_preds=pp, data_sources=data_sources))

    print("\nRidge:")
    ridge_result = timed("ridge",
        fc.evaluate_and_log, Ridge(alpha=1.0),
        X_eng_tr, y_tr, X_eng_te, y_te,
        name="ridge", baseline_preds=pp, data_sources=data_sources)
    results.append(ridge_result)

    print("\nHGB (direct, NaN-tolerant):")
    hgb_kw = dict(
        max_iter=800, learning_rate=0.03, max_depth=6,
        min_samples_leaf=200, l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=40,
    )
    t0 = time.time()
    hgb_preds, hgb_model = fit_hgb_direct(hgb_kw, X_eng_tr, y_tr, X_eng_te)
    print(f"  [{time.time() - t0:5.1f}s] hgb_direct")
    hgb_result = make_result("hgb_direct", hgb_preds, y_te, pp, model=hgb_model)
    fc.log_run(hgb_result, data_sources=data_sources,
               train_index=X_eng_tr.index, test_index=X_eng_te.index,
               n_features=X_eng_tr.shape[1],
               extra={"nan_handling": "native_hgb_no_harness_mask"})
    results.append(hgb_result)

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
    hgb_resid_result = make_result(
        "hgb_persistence_residual", hgb_persist_resid, y_te, pp, model=hgb_resid,
    )
    fc.log_run(hgb_resid_result, data_sources=data_sources,
               train_index=X_eng_tr.index, test_index=X_eng_te.index,
               n_features=X_eng_tr.shape[1],
               extra={"target": "y - hsig_m_now (persistence residual)"})
    results.append(hgb_resid_result)

    ridge_p = ridge_result.predictions
    stack = np.vstack([ridge_p, hgb_persist_resid])
    ens = np.nanmean(stack, axis=0)
    ens_result = make_result("ensemble_ridge_hgbresid", ens, y_te, pp)
    fc.log_run(ens_result, data_sources=data_sources,
               train_index=X_eng_tr.index, test_index=X_eng_te.index,
               n_features=X_eng_tr.shape[1],
               extra={"members": ["ridge", "hgb_persistence_residual"],
                      "combiner": "nanmean"})
    results.append(ens_result)

    print("\n=== Results (sorted by RMSE) ===")
    table = fc.compare(results).round(4)
    print(table.to_string())
    return table


if __name__ == "__main__":
    main()
