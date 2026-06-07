"""Per-horizon model search using ablation-recommended station sets.

Run:
    ./.venv/bin/python notebooks/recommended_sweep.py

For each forecast horizon, fits Ridge / HGB / GRU on each family's *recommended
station set* from ``feature_ablation`` and logs the result as
``recsweep_<family>_h<H>h``. Also fits a 3-family nanmean ensemble (Ridge +
HGB + GRU, each on its own recommended set) per horizon as
``recsweep_ensemble_h<H>h``.

The point: the ablation tells us *which* stations matter at each (horizon,
family); this script asks whether picking exactly that subset produces a
better model than either the primary-only baseline or the all-stations
ceiling already logged in ``experiments.jsonl``.

All runs share the same fixed window as the ablation (see
``forecast.ablation.load_all_sources``) and the same pinned 2023-01-01 test
cutoff. Persistence on this window is the skill baseline.
"""
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import forecast as fc
from forecast import ablation as ab
from forecast import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

HORIZONS_H: list[int] = [6, 12, 24, 36, 48, 72]
TEST_START = "2023-01-01"
RUN_PREFIX = "recsweep"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Mirror feature_ablation.py exactly so recsweep cells are directly comparable
# to ablate baseline/ceiling cells.
RIDGE_KW = {"alpha": 1.0}
HGB_KW = {
    "max_iter": 800, "learning_rate": 0.03, "max_depth": 6,
    "min_samples_leaf": 50, "l2_regularization": 1.0, "random_state": 42,
    "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 40,
}
GRU_CFG = {
    "seq_len": 48, "hidden": 64, "num_layers": 1,
    "epochs": 2, "weight_decay": 0.0, "rnn_dropout": 0.0,
    "lr": 1e-3, "batch_size": 512, "seed": 42, "scaler": "robust",
}


def _pinned_split(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    return int(idx.searchsorted(ts))


# ---------------------------------------------------------------------------
# Recommended-set lookup (re-runs the ablation report rule against the log)
# ---------------------------------------------------------------------------


def load_recommended_sets(threshold: float = 0.005) -> dict[tuple[str, int], list[str]]:
    """Read ``ablate_*`` rows from the log and apply the keep-if-helps rule."""
    log = fc.find_runs(name_prefix="ablate_")
    if log.empty:
        raise RuntimeError("No ablate_* rows in experiments.jsonl — run feature_ablation.py first.")
    df = pd.DataFrame({
        "family":    log["extra"].apply(lambda e: e.get("family")),
        "horizon_h": log["extra"].apply(lambda e: e.get("horizon_h")),
        "direction": log["extra"].apply(lambda e: e.get("direction")),
        "station":   log["extra"].apply(lambda e: e.get("station")),
        "RMSE":      log["metrics"].apply(lambda m: m.get("RMSE")),
        "timestamp": log["timestamp"],
    }).dropna(subset=["family", "horizon_h", "direction"])
    df["horizon_h"] = df["horizon_h"].astype(int)
    df = (df.sort_values("timestamp")
            .drop_duplicates(["family", "horizon_h", "direction", "station"], keep="last")
            .drop(columns="timestamp"))
    return ab.recommended_set(df, threshold=threshold)


# ---------------------------------------------------------------------------
# Per-family fit/predict cells — each returns (preds, metrics) and logs.
# Predictions are returned (in addition to metrics) so the ensemble can
# nanmean them at the end of each horizon.
# ---------------------------------------------------------------------------


def _ridge_cell(sources, h, stations, y_tr, y_te, pp, X_p_idx, extra_base):
    X = fc.build_design(*sources.subset(stations), kind="engineered")
    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos = _pinned_split(X.index, ts)
    X_tr, X_te = X.iloc[:pos], X.iloc[pos:]
    preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_tr)
    X_tr_imp, X_te_imp = preproc.transform(X_tr), preproc.transform(X_te)
    result = fc.evaluate_and_log(
        Ridge(**RIDGE_KW), X_tr_imp, y_tr, X_te_imp, y_te,
        name=f"{RUN_PREFIX}_ridge_h{h}h", baseline_preds=pp,
        data_sources=[fc.PRIMARY_BUOY] + stations,
        extra={**extra_base, "family": "ridge", "direction": "recommended",
               "station_set": stations, "n_features": int(X_tr_imp.shape[1])},
    )
    return result.predictions, result.metrics


def _hgb_cell(sources, h, stations, y_tr, y_te, pp,
              wave_level_tr, wave_level_te, extra_base):
    """HGB on persistence residual, matches feature_ablation."""
    X = fc.build_design(*sources.subset(stations), kind="engineered")
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
    preds = wave_level_te.to_numpy() + model.predict(X_te_raw.to_numpy())
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=pp)
    fc.log_run(
        fc.EvaluationResult(name=f"{RUN_PREFIX}_hgb_h{h}h", metrics=metrics,
                            predictions=preds, model=model),
        data_sources=[fc.PRIMARY_BUOY] + stations,
        train_index=X_tr.index, test_index=X_te.index,
        n_features=int(X_tr_raw.shape[1]),
        extra={**extra_base, "family": "hgb", "direction": "recommended",
               "station_set": stations, "n_features": int(X_tr_raw.shape[1]),
               "hgb_mode": "persistence_residual"},
    )
    return preds, metrics


def _gru_cell(sources, h, stations, y_tr, y_te, pp, extra_base):
    X = fc.build_design(*sources.subset(stations), kind="raw")
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
        scaler=GRU_CFG["scaler"], weight_decay=GRU_CFG["weight_decay"],
        rnn_dropout=GRU_CFG["rnn_dropout"],
    )
    t0 = time.time()
    model.fit(X_tr_imp, y_tr)
    preds = model.predict(X_te_imp)
    elapsed = time.time() - t0
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=pp)
    fc.log_run(
        fc.EvaluationResult(name=f"{RUN_PREFIX}_gru_h{h}h", metrics=metrics,
                            predictions=preds, model=model),
        data_sources=[fc.PRIMARY_BUOY] + stations,
        train_index=X_tr.index, test_index=X_te.index,
        n_features=int(X_tr_imp.shape[1]),
        extra={**extra_base, "family": "gru", "direction": "recommended",
               "station_set": stations, "n_features": int(X_tr_imp.shape[1]),
               "device": device, "elapsed_min": round(elapsed / 60, 2),
               **{k: v for k, v in GRU_CFG.items() if k != "seed"}},
    )
    return preds, metrics


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    print("Loading sources …")
    sources = fc.load_all_sources()
    print(f"  fixed window: {sources.window_start} → {sources.window_end}")
    print(f"  wave rows: {len(sources.wave):,}")

    recs = load_recommended_sets()
    print("\nRecommended sets (from ablate_* log):")
    for (fam, h), st in sorted(recs.items()):
        print(f"  {fam:>5}  h={h:>3}h  ({len(st):>2})  {st}")

    summary_rows: list[dict] = []
    for h in HORIZONS_H:
        print(f"\n=== h={h}h ===")
        horizon_steps = fc.hours_to_steps(h)
        y = fc.make_target(sources.wave, horizon_steps=horizon_steps)
        ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
        pos = _pinned_split(sources.wave.index, ts)
        y_tr, y_te = y.iloc[:pos], y.iloc[pos:]
        X_p = sources.wave[["hsig_m"]]
        X_p_tr, X_p_te = X_p.iloc[:pos], X_p.iloc[pos:]
        persist = fc.evaluate(
            fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te,
            name=f"persistence_h{h}h",
        )
        pp = persist.predictions
        wave_level_tr = sources.wave["hsig_m"].iloc[:pos]
        wave_level_te = sources.wave["hsig_m"].iloc[pos:]

        extra_base = {
            "horizon_h": h, "horizon_steps": horizon_steps,
            "ablation_window_start": sources.window_start.isoformat(),
            "ablation_window_end": sources.window_end.isoformat(),
            "imputation": "mean", "scaling": "robust",
            "selection_threshold_pct": 0.5,
        }

        family_preds: dict[str, np.ndarray] = {}
        for family, fn_args in [
            ("ridge", lambda s: _ridge_cell(sources, h, s, y_tr, y_te, pp, X_p_tr.index, extra_base)),
            ("hgb",   lambda s: _hgb_cell(sources, h, s, y_tr, y_te, pp,
                                          wave_level_tr, wave_level_te, extra_base)),
            ("gru",   lambda s: _gru_cell(sources, h, s, y_tr, y_te, pp, extra_base)),
        ]:
            stations = recs.get((family, h), [])
            t_cell = time.time()
            preds, metrics = fn_args(stations)
            family_preds[family] = preds
            print(f"  [{time.time()-t_cell:6.1f}s] {family:>5} ({len(stations):>2} st.)  "
                  f"RMSE {metrics['RMSE']:.4f}  Skill {metrics.get('SkillVsBaseline', 0):+.4f}")
            summary_rows.append({"horizon_h": h, "family": family,
                                 "RMSE": metrics["RMSE"], "skill": metrics.get("SkillVsBaseline")})

        # 3-family ensemble: nanmean over the per-family recommended-set preds.
        ens_preds = np.nanmean(np.vstack(list(family_preds.values())), axis=0)
        ens_metrics = summarise(y_te.to_numpy(), ens_preds, y_pred_baseline=pp)
        fc.log_run(
            fc.EvaluationResult(name=f"{RUN_PREFIX}_ensemble_h{h}h",
                                metrics=ens_metrics, predictions=ens_preds, model=None),
            data_sources=[fc.PRIMARY_BUOY] + sorted({s for stations in
                          [recs.get((f, h), []) for f in ("ridge","hgb","gru")] for s in stations}),
            train_index=y_tr.index, test_index=y_te.index, n_features=None,
            model_class="NanMeanEnsemble",
            extra={**extra_base, "family": "ensemble", "direction": "recommended",
                   "members": list(family_preds.keys()), "combiner": "nanmean"},
        )
        print(f"           ensemble       RMSE {ens_metrics['RMSE']:.4f}  "
              f"Skill {ens_metrics.get('SkillVsBaseline', 0):+.4f}")
        summary_rows.append({"horizon_h": h, "family": "ensemble",
                             "RMSE": ens_metrics["RMSE"], "skill": ens_metrics.get("SkillVsBaseline")})

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    summary = pd.DataFrame(summary_rows)
    print("\n=== Per-horizon RMSE (recommended sets) ===")
    print(summary.pivot(index="horizon_h", columns="family", values="RMSE").to_string())
    save_chart(summary, sources)


def save_chart(summary: pd.DataFrame, sources) -> None:
    """Per-horizon RMSE line chart for the 3 families + ensemble + persistence."""
    # Persistence on the same fixed window — anchor for skill scoring.
    persist_rmse = []
    for h in HORIZONS_H:
        horizon_steps = fc.hours_to_steps(h)
        y = fc.make_target(sources.wave, horizon_steps=horizon_steps)
        ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
        pos = _pinned_split(sources.wave.index, ts)
        y_tr, y_te = y.iloc[:pos], y.iloc[pos:]
        X_p_tr = sources.wave[["hsig_m"]].iloc[:pos]
        X_p_te = sources.wave[["hsig_m"]].iloc[pos:]
        res = fc.evaluate(fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te)
        persist_rmse.append(res.metrics["RMSE"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"ridge": "#1f77b4", "hgb": "#ff7f0e", "gru": "#2ca02c",
              "ensemble": "#d62728"}
    markers = {"ridge": "o", "hgb": "s", "gru": "^", "ensemble": "D"}
    for family in ["ridge", "hgb", "gru", "ensemble"]:
        sub = summary[summary["family"] == family].sort_values("horizon_h")
        lw = 2.5 if family == "ensemble" else 1.4
        ax.plot(sub["horizon_h"], sub["RMSE"],
                color=colors[family], marker=markers[family],
                markersize=8, linewidth=lw,
                label=f"{family} (recommended set)")
    ax.plot(HORIZONS_H, persist_rmse, color="#000", linestyle="--",
            linewidth=1.2, alpha=0.7, label="persistence", marker="")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("RMSE (m)")
    ax.set_xticks(HORIZONS_H)
    ax.set_title(
        "Per-horizon model search with ablation-recommended station sets\n"
        f"fixed window {sources.window_start.date()} → {sources.window_end.date()}  ·  "
        f"pinned test from {TEST_START}",
        fontsize=11,
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "recommended_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
