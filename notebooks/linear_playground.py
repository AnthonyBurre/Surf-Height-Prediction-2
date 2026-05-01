"""Linear / tree model playground for +12h hsig_m.

Run:  ./.venv/bin/python notebooks/linear_playground.py

A single CONFIG dict at the top controls everything: data window, neighbour
buoys, wind stations, FeatureConfig knobs, which models to run (Ridge / Lasso /
HGB), HGB residual-target mode, and an optional nanmean ensemble. Edit it in
place and re-run — no other code changes needed.

Each completed run appends to experiments.jsonl with a name derived from the
CONFIG, so back-to-back runs stay distinguishable.


Tips
----
- Set a model key to False to skip it, True to use the defaults below, or a
  dict to override specific hyperparameters (merged with the defaults).
- ``hgb_residual_target=True`` trains HGB on y - persistence(y) then adds
  persistence back at predict time; often improves HGB skill slightly.
- ``ensemble=True`` computes a nanmean of every enabled model's predictions.
- Year trimming is done BEFORE the wind overlap restriction, so
  ``year_max=2024`` + a non-empty ``wind_stations`` just restricts to the
  selected stations' wind window (currently 2015-2024 for both).
- Persistence is computed on the same test window as all other models.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge

import forecast as fc
from forecast.metrics import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

# ---------------------------------------------------------------------------
# CONFIG — edit me
# ---------------------------------------------------------------------------

CONFIG: dict = {
    # --- data window ---
    "year_min":     None,   # None = data start; e.g. 2024 for the short multi-buoy window
    "year_max":     2024,   # None = data end;   e.g. 2024 to match the wind window

    # --- extra sources ---
    "neighbours":     ["caloundra", "brisbane", "goldcoast", "north-moreton-bay"],     # any subset of: "brisbane", "caloundra", "goldcoast", "north-moreton-bay"
    "wind_stations":  ["mountain-creek", "deception-bay"],  # any subset of: "mountain-creek", "deception-bay"; [] disables wind

    # --- feature engineering (FeatureConfig knobs) ---
    # Set to None to use the package defaults.
    "lag_steps":              None,   # default: [1,2,3,6,12,24,48,96,144]
    "roll_windows":           None,   # default: [12,24,48,96]
    "delta_steps":            None,   # default: [6,12,24,48]
    "neighbour_lag_steps":    None,   # default: [1,2,3,6,12,24]
    "neighbour_roll_windows": None,   # default: [6,12,24]

    # --- models ---
    # True  → run with default hyperparams shown below
    # dict  → run, merging the dict over the defaults
    # False → skip
    "ridge":  True,     # default: {"alpha": 1.0}
    "lasso":  True,    # default: {"alpha": 0.001, "max_iter": 10000}
    "hgb":    True,    # default: see HGB_DEFAULTS below

    # HGB-specific: train on y − persistence(y) instead of y
    "hgb_residual_target": True,

    # nanmean ensemble of all enabled model predictions
    "ensemble": True,

    # --- run / logging ---
    "run_name":     "lineopt",  # log entries get suffixed with active models + sources
    "log_to_jsonl": True,
}

# ---------------------------------------------------------------------------
# Model defaults — merged with any dict provided in CONFIG
# ---------------------------------------------------------------------------

_RIDGE_DEFAULTS  = {"alpha": 1.0}
_LASSO_DEFAULTS  = {"alpha": 0.001, "max_iter": 10000}
_HGB_DEFAULTS    = {
    "max_iter": 800, "learning_rate": 0.03, "max_depth": 6,
    "min_samples_leaf": 50, "l2_regularization": 1.0, "random_state": 42,
    "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 40,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_hyperparams(cfg_value: bool | dict, defaults: dict) -> dict | None:
    """Return merged hyperparams, or None if the model is disabled."""
    if cfg_value is False:
        return None
    if cfg_value is True:
        return defaults.copy()
    return {**defaults, **cfg_value}


def _timed(label: str, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    print(f"  [{time.time() - t0:5.1f}s] {label}")
    return out


def _mean_impute(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = SimpleImputer(strategy="mean")
    return (
        pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index),
        pd.DataFrame(imp.transform(X_te),     columns=X_te.columns, index=X_te.index),
    )


def _make_result(name: str, preds: np.ndarray, y_te: pd.Series,
                 baseline_preds: np.ndarray, model=None) -> fc.EvaluationResult:
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=baseline_preds)
    return fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=model)


def _wind_tag(stations: list[str]) -> str:
    """Two-letter abbreviation per station (e.g. mountain-creek -> mc)."""
    return "+".join("".join(part[0] for part in s.split("-")) for s in stations)


def _make_run_name(cfg: dict) -> str:
    parts = [cfg["run_name"]]
    if cfg["wind_stations"]:
        parts.append("wind-" + _wind_tag(cfg["wind_stations"]))
    if cfg["neighbours"]:
        parts.append("+".join(n[:4] for n in cfg["neighbours"]))
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_wave(year_min, year_max) -> pd.DataFrame:
    df = fc.load_data()
    if year_min is not None:
        df = df.loc[df.index.year >= year_min]
    if year_max is not None:
        df = df.loc[df.index.year <= year_max]
    return df


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def _build_features(wave: pd.DataFrame,
                    neighbour_series: dict[str, pd.Series],
                    wind: pd.DataFrame | None,
                    cfg: dict) -> pd.DataFrame:
    fc_kwargs: dict = {}
    for key in ("lag_steps", "roll_windows", "delta_steps",
                "neighbour_lag_steps", "neighbour_roll_windows"):
        if cfg.get(key) is not None:
            fc_kwargs[key] = cfg[key]
    feat_cfg = fc.FeatureConfig(**fc_kwargs) if fc_kwargs else None

    merged = wave.copy()
    neighbour_cols = []
    for name, series in neighbour_series.items():
        col = f"{name}_hsig_m"
        merged[col] = series
        neighbour_cols.append(col)

    mool_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
    X = fc.build_mooloolaba_features(mool_only, config=feat_cfg)

    if neighbour_cols:
        X = fc.add_neighbour_features(X, merged, neighbour_cols, config=feat_cfg)

    if wind is not None:
        wind_cols = [c for c in wind.columns if not c.endswith("_deg")]
        X = fc.add_neighbour_features(X, wind, wind_cols, config=feat_cfg)

    return X


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _run_ridge(kw, X_tr, y_tr, X_te, y_te, pp, name, sources, extra, log) -> fc.EvaluationResult:
    print(f"\n=== Ridge  alpha={kw['alpha']} ===")
    return _timed(name, fc.evaluate_and_log, Ridge(**kw),
                  X_tr, y_tr, X_te, y_te,
                  name=name, baseline_preds=pp,
                  data_sources=sources, extra=extra) if log else \
           _timed(name, fc.evaluate, Ridge(**kw),
                  X_tr, y_tr, X_te, y_te, baseline_preds=pp)


def _run_lasso(kw, X_tr, y_tr, X_te, y_te, pp, name, sources, extra, log) -> fc.EvaluationResult:
    print(f"\n=== Lasso  alpha={kw['alpha']} ===")
    result = _timed(name, fc.evaluate_and_log, Lasso(**kw),
                    X_tr, y_tr, X_te, y_te,
                    name=name, baseline_preds=pp,
                    data_sources=sources, extra=extra) if log else \
             _timed(name, fc.evaluate, Lasso(**kw),
                    X_tr, y_tr, X_te, y_te, baseline_preds=pp)
    coef = result.model.coef_
    n_nonzero = int((coef != 0).sum())
    print(f"  Non-zero coefficients: {n_nonzero} / {len(coef)}")
    top = (pd.Series(np.abs(coef), index=X_tr.columns)
           .sort_values(ascending=False).head(10))
    print("  Top 10 by |coef|:")
    for feat, val in top.items():
        print(f"    {feat:42s}  {val:.4f}")
    return result


def _run_hgb(kw, X_tr, y_tr, X_te, y_te, pp, wave_tr, wave_te,
             residual_target, name, sources, extra, log) -> fc.EvaluationResult:
    mode = "persistence_residual" if residual_target else "direct"
    print(f"\n=== HGB  ({mode}) ===")

    if residual_target:
        hsig_now_tr = wave_tr.to_numpy()
        hsig_now_te = wave_te.to_numpy()
        y_res = y_tr.to_numpy() - hsig_now_tr
        mask = np.isfinite(y_res)
        m = HistGradientBoostingRegressor(**kw)
        t0 = time.time()
        m.fit(X_tr.loc[mask].to_numpy(), y_res[mask])
        preds = hsig_now_te + m.predict(X_te.to_numpy())
        print(f"  [{time.time() - t0:5.1f}s] {name}")
    else:
        mask = ~y_tr.isna()
        m = HistGradientBoostingRegressor(**kw)
        t0 = time.time()
        m.fit(X_tr.loc[mask].to_numpy(), y_tr.loc[mask].to_numpy())
        preds = m.predict(X_te.to_numpy())
        print(f"  [{time.time() - t0:5.1f}s] {name}")

    result = _make_result(name, preds, y_te, pp, model=m)
    if log:
        fc.log_run(result, data_sources=sources,
                   train_index=X_tr.index, test_index=X_te.index,
                   n_features=X_tr.shape[1],
                   extra={**extra, "hgb_mode": mode, "nan_handling": "native_hgb"})

    if hasattr(m, "feature_importances_"):
        top_imp = (pd.Series(m.feature_importances_, index=X_tr.columns)
                   .sort_values(ascending=False).head(10))
        print("  Top 10 features by importance:")
        for feat, val in top_imp.items():
            print(f"    {feat:42s}  {val:.4f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG

    ridge_kw = _resolve_hyperparams(cfg["ridge"], _RIDGE_DEFAULTS)
    lasso_kw = _resolve_hyperparams(cfg["lasso"], _LASSO_DEFAULTS)
    hgb_kw   = _resolve_hyperparams(cfg["hgb"],   _HGB_DEFAULTS)

    wave = _load_wave(cfg["year_min"], cfg["year_max"])
    neighbours = fc.load_neighbours(wave.index, cfg["neighbours"])
    wind = fc.load_wind(wave.index, cfg["wind_stations"])

    wave, neighbours, wind = fc.restrict_to_overlap(wave, neighbours, wind)

    print(f"window         : {wave.index.min().date()} → {wave.index.max().date()}")
    print(f"rows           : {len(wave):,}")

    X = _build_features(wave, neighbours, wind, cfg)
    y = fc.make_target(wave)
    X_p = wave[["hsig_m"]]

    X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y)
    X_p_tr, X_p_te, _, _   = fc.chronological_split(X_p, y)
    X_tr_imp, X_te_imp     = _mean_impute(X_tr, X_te)

    nan_pct = X.isna().mean().mul(100)
    worst = nan_pct.sort_values(ascending=False).head(3).round(2).to_dict()
    print(f"features       : {X.shape[1]}  |  train: {len(X_tr):,}  test: {len(X_te):,}")
    print(f"top NaN cols   : {worst}\n")

    run_name = _make_run_name(cfg)
    sources = ["mooloolaba"] + cfg["neighbours"] + cfg["wind_stations"]
    log = cfg["log_to_jsonl"]
    window_str = f"{wave.index.min().date()}:{wave.index.max().date()}"
    extra_base: dict = {
        "window": window_str,
        "imputation": "mean",
        "n_neighbours": len(cfg["neighbours"]),
        "wind_stations": cfg["wind_stations"],
    }

    print("=== Persistence baseline ===")
    persist_name = f"{run_name}_persistence"
    persist = _timed(persist_name, fc.evaluate_and_log if log else fc.evaluate,
                     fc.PersistenceForecaster(),
                     X_p_tr, y_tr, X_p_te, y_te,
                     name=persist_name, data_sources=["mooloolaba"],
                     extra={"window": window_str}) if log else \
              _timed(persist_name, fc.evaluate,
                     fc.PersistenceForecaster(),
                     X_p_tr, y_tr, X_p_te, y_te)
    pp = persist.predictions
    print(f"persistence    : RMSE {persist.metrics['RMSE']:.4f}\n")

    results: list[fc.EvaluationResult] = []

    if ridge_kw is not None:
        r = _run_ridge(ridge_kw, X_tr_imp, y_tr, X_te_imp, y_te, pp,
                       f"{run_name}_ridge", sources,
                       {**extra_base, **ridge_kw}, log)
        results.append(r)

    if lasso_kw is not None:
        r = _run_lasso(lasso_kw, X_tr_imp, y_tr, X_te_imp, y_te, pp,
                       f"{run_name}_lasso", sources,
                       {**extra_base, **lasso_kw}, log)
        results.append(r)

    if hgb_kw is not None:
        wave_tr = wave["hsig_m"].loc[X_tr.index]
        wave_te = wave["hsig_m"].loc[X_te.index]
        r = _run_hgb(hgb_kw, X_tr, y_tr, X_te, y_te, pp,
                     wave_tr, wave_te,
                     cfg.get("hgb_residual_target", False),
                     f"{run_name}_hgb", sources,
                     extra_base, log)
        results.append(r)

    if cfg.get("ensemble") and len(results) >= 2:
        ens_preds = np.nanmean(np.vstack([r.predictions for r in results]), axis=0)
        ens_name  = f"{run_name}_ensemble"
        ens = _make_result(ens_name, ens_preds, y_te, pp)
        if log:
            fc.log_run(ens, data_sources=sources,
                       train_index=X_tr.index, test_index=X_te.index,
                       n_features=X.shape[1],
                       model_class="NanMeanEnsemble",
                       extra={**extra_base,
                              "members": [r.name for r in results],
                              "combiner": "nanmean"})
        results.append(ens)

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"{'='*60}")
    all_results = [persist] + results
    table = fc.compare(all_results).round(4)
    print(table.to_string())

    if log:
        log_df = fc.read_log()
        recent = (log_df[log_df["name"].str.startswith(cfg["run_name"])]
                  .sort_values("timestamp").tail(10))
        if len(recent) > 1:
            print("\nrecent playground runs (most recent last):")
            for _, row in recent.iterrows():
                m = row["metrics"]
                ex = row.get("extra") or {}
                ws = ex.get("wind_stations", "?")
                tag = f"w={ws} nb={ex.get('n_neighbours','?')}"
                skill = m.get("SkillVsBaseline", 0)
                print(f"  {row['name']:60s}  {tag:20s}  RMSE {m['RMSE']:.4f}  Skill {skill:+.4f}")


if __name__ == "__main__":
    main()
