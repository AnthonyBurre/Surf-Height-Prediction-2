"""Linear / tree model playground for +12h hsig_m.

Run:  ./.venv/bin/python notebooks/linear_playground.py

A single CONFIG dict at the top controls everything: data window, neighbour
buoys, wind on/off, FeatureConfig knobs, which models to run (Ridge / Lasso /
HGB), HGB residual-target mode, and an optional nanmean ensemble. Edit it in
place and re-run — no other code changes needed.

Each completed run appends to experiments.jsonl with a name derived from the
CONFIG, so back-to-back runs stay distinguishable.

Why this script exists
----------------------
Four earlier scripts cover overlapping but hard-coded scenarios:

  forecast_v2.py               Mooloolaba 2015-2025, Ridge + HGB variants + ensemble
  mooloolaba_brisbane_forecast.py  Ridge + Lasso with Brisbane, 2015-2025
  mooloolaba_wind_forecast.py  Ridge + Lasso with Mountain Creek wind, 2015-2024
  multi_buoy_forecast.py       Ridge + HGB, 2024-2025 with 3 neighbour buoys

This playground replaces all four. Paste any of the starting points below to
reproduce a previous run, or tweak freely.

Starting points (paste into CONFIG)
------------------------------------
1. Full-history Mooloolaba Ridge (replicates forecast_v2):
     year_min=None, year_max=None, neighbours=[], include_wind=False,
     ridge=True, hgb=True, hgb_residual_target=True, ensemble=True

2. Mooloolaba + Brisbane linear (replicates mooloolaba_brisbane_forecast):
     year_min=None, year_max=None, neighbours=["brisbane"],
     ridge=True, lasso={"alpha": 0.001, "max_iter": 5000}

3. Wind-augmented window (replicates mooloolaba_wind_forecast):
     year_min=None, year_max=2024, include_wind=True,
     ridge=True, lasso={"alpha": 0.001, "max_iter": 10000}

4. Multi-buoy 2024-2025 (replicates multi_buoy_forecast):
     year_min=2024, year_max=None,
     neighbours=["caloundra", "brisbane", "goldcoast"],
     ridge=True, hgb=True

5. Kitchen sink — everything:
     neighbours=["brisbane", "caloundra", "goldcoast", "north-moreton-bay"],
     include_wind=True, year_max=2024,
     ridge=True, lasso=True, hgb=True, hgb_residual_target=True, ensemble=True

Tips
----
- Set a model key to False to skip it, True to use the defaults below, or a
  dict to override specific hyperparameters (merged with the defaults).
- ``hgb_residual_target=True`` trains HGB on y − persistence(y) then adds
  persistence back at predict time; often improves HGB skill slightly.
- ``ensemble=True`` computes a nanmean of every enabled model's predictions.
- Year trimming is done BEFORE the wind overlap restriction, so
  ``year_max=2024`` + ``include_wind=True`` just restricts to 2015-2024 wind
  data as usual.
- Persistence is computed on the same test window as all other models.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge

import forecast as fc
from forecast.features import encode_circular
from forecast.metrics import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# CONFIG — edit me
# ---------------------------------------------------------------------------

CONFIG: dict = {
    # --- data window ---
    "year_min":     None,   # None = data start; e.g. 2024 for the short multi-buoy window
    "year_max":     None,   # None = data end;   e.g. 2024 to match the wind window

    # --- extra sources ---
    "neighbours":   [],     # any subset of: "brisbane", "caloundra", "goldcoast", "north-moreton-bay"
    "include_wind": False,  # adds Mountain Creek hourly wind (2015-2024)

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
    "lasso":  False,    # default: {"alpha": 0.001, "max_iter": 10000}
    "hgb":    False,    # default: see HGB_DEFAULTS below

    # HGB-specific: train on y − persistence(y) instead of y
    "hgb_residual_target": False,

    # nanmean ensemble of all enabled model predictions
    "ensemble": False,

    # --- run / logging ---
    "run_name":     "linplay",  # log entries get suffixed with active models + sources
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

_NEIGHBOUR_FILES = {
    "brisbane":          "brisbane_wave_data_2015-2025.csv",
    "caloundra":         "caloundra_wave_data_2013-2025.csv",
    "goldcoast":         "gold-coast_wave_data_2015-2025.csv",
    "north-moreton-bay": "north-moreton-bay_wave_data_2010-2025.csv",
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


def _make_run_name(cfg: dict) -> str:
    parts = [cfg["run_name"]]
    active = [m for m in ("ridge", "lasso", "hgb") if cfg[m] is not False]
    parts.extend(active)
    if cfg["include_wind"]:
        parts.append("wind")
    if cfg["neighbours"]:
        parts.append("+".join(n[:4] for n in cfg["neighbours"]))
    if cfg.get("hgb_residual_target") and cfg["hgb"] is not False:
        parts.append("resid")
    if cfg.get("ensemble"):
        parts.append("ens")
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


def _load_neighbours(target_index: pd.DatetimeIndex,
                     neighbours: list[str]) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for name in neighbours:
        if name not in _NEIGHBOUR_FILES:
            raise ValueError(f"Unknown neighbour {name!r}; supported: {list(_NEIGHBOUR_FILES)}")
        nb = pd.read_csv(
            DATA_DIR / _NEIGHBOUR_FILES[name],
            parse_dates=["datetime_utc"], index_col="datetime_utc",
        )
        out[name] = nb["hsig_m"].reindex(target_index)
    return out


def _load_wind(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    wind = pd.read_csv(
        DATA_DIR / "mountain-creek_wind_data_2015-2024.csv",
        parse_dates=["datetime_utc"], index_col="datetime_utc",
    )
    wind = encode_circular(wind, columns=["wind_dir_deg"])
    return wind.reindex(target_index, method="ffill")


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
    neighbours = _load_neighbours(wave.index, cfg["neighbours"])
    wind = _load_wind(wave.index) if cfg["include_wind"] else None

    if wind is not None:
        valid = wind.dropna(how="all")
        start, end = valid.index.min(), valid.index.max()
        wave = wave.loc[start:end]
        neighbours = {k: v.loc[start:end] for k, v in neighbours.items()}
        wind = wind.loc[start:end]
    elif cfg["neighbours"]:
        starts = [s.dropna().index.min() for s in neighbours.values()]
        ends   = [s.dropna().index.max() for s in neighbours.values()]
        start, end = max(starts), min(ends)
        wave = wave.loc[start:end]
        neighbours = {k: v.loc[start:end] for k, v in neighbours.items()}

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
    sources = ["mooloolaba"] + cfg["neighbours"] + (["mountain-creek"] if cfg["include_wind"] else [])
    log = cfg["log_to_jsonl"]
    window_str = f"{wave.index.min().date()}:{wave.index.max().date()}"
    extra_base: dict = {
        "window": window_str,
        "imputation": "mean",
        "n_neighbours": len(cfg["neighbours"]),
        "include_wind": cfg["include_wind"],
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
                tag = f"w={ex.get('include_wind','?')} nb={ex.get('n_neighbours','?')}"
                skill = m.get("SkillVsBaseline", 0)
                print(f"  {row['name']:60s}  {tag:20s}  RMSE {m['RMSE']:.4f}  Skill {skill:+.4f}")


if __name__ == "__main__":
    main()
