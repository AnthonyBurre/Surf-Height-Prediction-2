"""Horizon sweep — how each (architecture, source) combo performs across horizons.

Run:  ./.venv/bin/python notebooks/horizon_sweep.py

Loops six forecast horizons (6h, 12h, 24h, 36h, 48h, 72h) over four source
combos that came out of the linear sweep:

  - solo      — mooloolaba only, no neighbours, no wind  (2015-2024)
  - tweed_mc  — mooloolaba + tweed-heads + mountain-creek wind  (2015-2024)
  - baseline  — mooloolaba + 5 neighbour buoys + 3 wind stations  (2015-2024)
  - wide      — mooloolaba + 7 neighbour buoys + 4 wind stations  (2019-2024)

For each combo the feature matrix is built ONCE (horizon-independent); only
the target column and `evaluate` call are redone per horizon. Models: Ridge,
Lasso, HGB-on-persistence-residual, and a nanmean ensemble of the three.

All runs share the same pinned test-window cutoff (2023-01-01 AEST) so skill
scores at any one horizon are directly comparable across combos. Persistence
is computed per-horizon (it varies a lot — the autocorrelation collapses).

Saves:
  horizon_sweep.png — two stacked panels (skill-vs-persistence and RMSE)
                       with one line per combo's ensemble plus persistence.
  experiments.jsonl entries under name 'hsweep_<combo>_h<H>h_<model>'.

Sequence models are intentionally out of scope for the first pass; they'd
either need per-horizon retraining (~2 min each) or the multi-output-head
refactor of `neural.py` discussed in the README's Future directions.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge

import forecast as fc
from forecast.metrics import summarise

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

HORIZONS_H: list[int] = [6, 12, 24, 36, 48, 72]
TEST_START = "2023-01-01"

COMBOS: list[dict] = [
    {
        "name": "solo",
        "year_min": None, "year_max": 2024,
        "neighbours": [],
        "wind":       [],
    },
    {
        "name": "tweed_mc",
        "year_min": None, "year_max": 2024,
        "neighbours": ["tweed-heads"],
        "wind":       ["mountain-creek"],
    },
    {
        "name": "baseline",
        "year_min": None, "year_max": 2024,
        "neighbours": ["caloundra", "brisbane", "gold-coast", "north-moreton-bay", "tweed-heads"],
        "wind":       ["mountain-creek", "deception-bay", "lytton"],
    },
    {
        "name": "wide",
        "year_min": 2019, "year_max": 2024,
        "neighbours": ["caloundra", "brisbane", "gold-coast", "north-moreton-bay",
                       "palm-beach", "tweed-heads", "wide-bay"],
        "wind":       ["mountain-creek", "deception-bay", "lytton", "southport"],
    },
]

COMBO_COLORS = {
    "solo":     "#1f77b4",
    "tweed_mc": "#ff7f0e",
    "baseline": "#2ca02c",
    "wide":     "#d62728",
}

# Two baselines plotted as gray reference lines on every architecture panel.
# Persistence is the project's headline baseline (skill ≡ 0 by definition);
# climatology-hour is the "regress to the diurnal mean" floor that any real
# model must beat. Naming them here keeps the plot labels honest.
BASELINE_STYLES = {
    "persistence":      {"linestyle": "-",  "color": "#000000", "linewidth": 1.6, "alpha": 0.9},
    "climatology_hour": {"linestyle": ":",  "color": "#555555", "linewidth": 1.5, "alpha": 0.85},
}
ARCHITECTURES = ["ridge", "lasso", "hgb"]

# Hyperparams — kept exactly in step with the linear playground defaults
# so the h=12 row of this sweep reproduces the v1 baseline numbers in the
# README's "Linear models" table.
RIDGE_KW = {"alpha": 1.0}
LASSO_KW = {"alpha": 0.001, "max_iter": 10000}
HGB_KW   = {
    "max_iter": 800, "learning_rate": 0.03, "max_depth": 6,
    "min_samples_leaf": 50, "l2_regularization": 1.0, "random_state": 42,
    "early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 40,
}


# ---------------------------------------------------------------------------
# Feature build (horizon-independent — done once per combo)
# ---------------------------------------------------------------------------

def build_combo(combo: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (wave_df, X_features, source_labels) for a combo."""
    wave = fc.restrict_to_years(
        fc.load_data(buoy="mooloolaba"), combo["year_min"], combo["year_max"],
    )
    neighbours = fc.load_neighbours(wave.index, combo["neighbours"])
    wind = fc.load_wind(wave.index, combo["wind"])
    wave, neighbours, wind = fc.restrict_to_overlap(wave, neighbours, wind)

    merged, neighbour_cols, _ = fc.assemble_inputs(wave, neighbours, wind)
    primary_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
    X = fc.build_buoy_features(primary_only)
    if neighbour_cols:
        X = fc.add_neighbour_features(X, merged, neighbour_cols)
    if wind is not None:
        wind_cols = [c for c in wind.columns if not c.endswith("_deg")]
        X = fc.add_neighbour_features(X, wind, wind_cols)

    sources = ["mooloolaba"] + combo["neighbours"] + combo["wind"]
    return wave, X, sources


# ---------------------------------------------------------------------------
# Non-ML baselines — Mooloolaba-only, combo-independent
# ---------------------------------------------------------------------------

def compute_baselines(wave_full: pd.DataFrame, horizon_h: int) -> dict[str, dict[str, float]]:
    """Persistence + ClimatologyHour at one horizon.

    Both only need the Mooloolaba ``hsig_m`` column, so they're computed
    once per horizon (not once per combo). Returned skill is vs Persistence.
    """
    horizon_steps = horizon_h * 2
    y = fc.make_target(wave_full, horizon_steps=horizon_steps)
    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos_w = _pinned_split(wave_full.index, ts)
    X_p_tr = wave_full[["hsig_m"]].iloc[:pos_w]
    X_p_te = wave_full[["hsig_m"]].iloc[pos_w:]
    y_tr, y_te = y.iloc[:pos_w], y.iloc[pos_w:]

    # Persistence — the reference for climatology's skill score
    persist = fc.evaluate(
        fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te,
        name=f"persistence_h{horizon_h}h",
    )
    pp = persist.predictions

    # ClimatologyHour — learn hour-of-target-time mean from train
    ch = fc.ClimatologyHourForecaster(horizon_steps=horizon_steps)
    ch.fit(X_p_tr, y_tr)
    ch_preds = ch.predict(X_p_te)

    return {
        "persistence":      {**persist.metrics, "SkillVsBaseline": 0.0},
        "climatology_hour": summarise(y_te.to_numpy(), ch_preds, y_pred_baseline=pp),
    }


# ---------------------------------------------------------------------------
# One (combo, horizon) cell — fit/predict/score/log every model
# ---------------------------------------------------------------------------

def _pinned_split(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    return int(idx.searchsorted(ts))


def run_cell(
    combo: dict,
    horizon_h: int,
    wave: pd.DataFrame,
    X: pd.DataFrame,
    sources: list[str],
    *,
    log: bool,
) -> dict[str, dict[str, float]]:
    """Fit ridge/lasso/hgb-residual/ensemble at one horizon. Return per-model metrics."""
    horizon_steps = horizon_h * 2  # 30-min cadence
    y = fc.make_target(wave, horizon_steps=horizon_steps)

    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos   = _pinned_split(X.index, ts)
    pos_w = _pinned_split(wave.index, ts)
    X_tr, X_te = X.iloc[:pos], X.iloc[pos:]
    y_tr, y_te = y.iloc[:pos], y.iloc[pos:]
    wave_tr = wave["hsig_m"].iloc[:pos_w]
    wave_te = wave["hsig_m"].iloc[pos_w:]
    X_p_tr = wave[["hsig_m"]].iloc[:pos_w]
    X_p_te = wave[["hsig_m"]].iloc[pos_w:]

    preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_tr)
    X_tr_imp = preproc.transform(X_tr)
    X_te_imp = preproc.transform(X_te)
    X_tr_raw = X_tr[preproc.kept_columns_]  # HGB-native NaN
    X_te_raw = X_te[preproc.kept_columns_]

    name_prefix = f"hsweep_{combo['name']}_h{horizon_h}h"
    window_str = f"{wave.index.min().date()}:{wave.index.max().date()}"
    extra = {
        "window": window_str, "imputation": "mean", "scaling": "robust",
        "horizon_h": horizon_h, "horizon_steps": horizon_steps,
        "combo": combo["name"], "n_neighbours": len(combo["neighbours"]),
        "wind_stations": combo["wind"],
    }

    # --- Persistence (per-horizon — gets worse as h grows) ---------------
    persist = fc.evaluate_and_log(
        fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name=f"{name_prefix}_persistence",
        data_sources=["mooloolaba"], extra={"window": window_str, "horizon_h": horizon_h},
        log=log,
    )
    pp = persist.predictions

    # --- Ridge ------------------------------------------------------------
    ridge = fc.evaluate_and_log(
        Ridge(**RIDGE_KW),
        X_tr_imp, y_tr, X_te_imp, y_te,
        name=f"{name_prefix}_ridge", baseline_preds=pp,
        data_sources=sources, extra=extra, log=log,
    )

    # --- Lasso ------------------------------------------------------------
    lasso = fc.evaluate_and_log(
        Lasso(**LASSO_KW),
        X_tr_imp, y_tr, X_te_imp, y_te,
        name=f"{name_prefix}_lasso", baseline_preds=pp,
        data_sources=sources, extra=extra, log=log,
    )

    # --- HGB on persistence residual --------------------------------------
    # y_residual = y(t+h) - y(t).  At long h this delta has more variance than
    # the level itself, which is exactly why letting HGB learn the delta beats
    # learning the level directly.
    y_res = y_tr - wave_tr
    res_mask = ~y_res.isna() & ~X_tr_raw.isna().any(axis=1)
    hgb_model = HistGradientBoostingRegressor(**HGB_KW)
    hgb_model.fit(X_tr_raw.loc[res_mask].to_numpy(), y_res.loc[res_mask].to_numpy())
    hgb_preds = wave_te.to_numpy() + hgb_model.predict(X_te_raw.to_numpy())
    hgb_metrics = summarise(y_te.to_numpy(), hgb_preds, y_pred_baseline=pp)
    hgb_result = fc.EvaluationResult(
        name=f"{name_prefix}_hgb", metrics=hgb_metrics,
        predictions=hgb_preds, model=hgb_model,
    )
    if log:
        fc.log_run(
            hgb_result, data_sources=sources,
            train_index=X_tr.index, test_index=X_te.index, n_features=X_tr.shape[1],
            extra={**extra, "hgb_mode": "persistence_residual", "nan_handling": "native_hgb"},
        )

    # --- Ensemble (nanmean of the three) ---------------------------------
    members = [ridge.predictions, lasso.predictions, hgb_preds]
    ens_preds = np.nanmean(np.vstack(members), axis=0)
    ens_metrics = summarise(y_te.to_numpy(), ens_preds, y_pred_baseline=pp)
    ens_result = fc.EvaluationResult(
        name=f"{name_prefix}_ensemble", metrics=ens_metrics,
        predictions=ens_preds, model=None,
    )
    if log:
        fc.log_run(
            ens_result, data_sources=sources,
            train_index=X_tr.index, test_index=X_te.index, n_features=X.shape[1],
            model_class="NanMeanEnsemble",
            extra={**extra, "members": [r.name for r in (ridge, lasso, hgb_result)],
                   "combiner": "nanmean"},
        )

    return {
        "persistence": persist.metrics,
        "ridge":       ridge.metrics,
        "lasso":       lasso.metrics,
        "hgb":         hgb_metrics,
        "ensemble":    ens_metrics,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sweep(
    results: dict[str, dict[int, dict[str, dict[str, float]]]],
    baselines: dict[int, dict[str, dict[str, float]]],
) -> None:
    """Small multiples by architecture: one column per Ridge/Lasso/HGB.

    Each column has two stacked panels — skill on top, RMSE on bottom —
    and within a panel the four combos are colored lines plus the three
    baselines as gray reference lines. Y-axes are shared across rows so
    combo crossings between architectures read off directly.

    Args:
        results:   results[combo_name][horizon_h][model_name] = metrics
        baselines: baselines[horizon_h][baseline_name]        = metrics
    """
    horizons = sorted(baselines.keys())
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), sharex="col", sharey="row")

    for col, arch in enumerate(ARCHITECTURES):
        ax_skill = axes[0, col]
        ax_rmse  = axes[1, col]

        # --- Combo lines (one per combo, colored) ------------------------
        for combo in COMBOS:
            cname = combo["name"]
            skills = [results[cname][h][arch]["SkillVsBaseline"] for h in horizons]
            rmses  = [results[cname][h][arch]["RMSE"]            for h in horizons]
            ax_skill.plot(horizons, skills, marker="o", linewidth=2.0, markersize=7,
                          color=COMBO_COLORS[cname], label=cname)
            ax_rmse.plot( horizons, rmses,  marker="o", linewidth=2.0, markersize=7,
                          color=COMBO_COLORS[cname], label=cname)

        # --- Baselines (three gray reference lines per panel) ------------
        for bname, style in BASELINE_STYLES.items():
            skills = [baselines[h][bname]["SkillVsBaseline"] for h in horizons]
            rmses  = [baselines[h][bname]["RMSE"]            for h in horizons]
            ax_skill.plot(horizons, skills, label=bname.replace("_", " "), **style)
            ax_rmse.plot( horizons, rmses,  label=bname.replace("_", " "), **style)

        ax_skill.axhline(0, color="#999999", linewidth=0.8, linestyle="-", alpha=0.6)
        # Climatology at h=6 sits around skill -1.7; the combo lines are in
        # 0–0.45. Pad below climatology's worst point so the line stays in
        # frame at every horizon.
        ax_skill.set_ylim(-1.9, 0.55)
        ax_skill.set_title(arch.upper(), fontsize=13, fontweight="bold")
        ax_skill.grid(alpha=0.3)
        ax_rmse.grid(alpha=0.3)
        ax_rmse.set_xticks(horizons)
        ax_rmse.set_xticklabels([f"{h}h" for h in horizons])
        ax_rmse.set_xlabel("Forecast horizon (hours)")

        if col == 0:
            ax_skill.set_ylabel("Skill vs persistence")
            ax_rmse.set_ylabel("RMSE (m)")

    # One legend to the right — keep panels uncluttered. Pull handles from
    # both rows so combo colors AND baseline linestyles end up in the legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right", bbox_to_anchor=(1.005, 0.5),
        fontsize=10, frameon=True,
        title="combos (colored)\nbaselines (gray)",
        title_fontsize=9,
    )

    fig.suptitle(
        "Multi-horizon performance by architecture × data-source combo  —  pinned 2023-01-01 → 2024-12-31 test window",
        fontsize=13, y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    out = FIG_DIR / "horizon_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    results: dict[str, dict[int, dict[str, dict[str, float]]]] = {}

    # --- Non-ML baselines first (cheap; combo-independent) ---------------
    wave_full = fc.restrict_to_years(fc.load_data(buoy="mooloolaba"), None, 2024)
    baselines: dict[int, dict[str, dict[str, float]]] = {}
    print(f"\n--- baselines (Mooloolaba {wave_full.index.min().date()} → {wave_full.index.max().date()}) ---")
    for h in HORIZONS_H:
        baselines[h] = compute_baselines(wave_full, h)
        b = baselines[h]
        print(
            f"  h={h:>3}h  "
            f"persistence RMSE {b['persistence']['RMSE']:.4f}  |  "
            f"climatology-hour  {b['climatology_hour']['RMSE']:.4f} (skill {b['climatology_hour']['SkillVsBaseline']:+.4f})"
        )

    for combo in COMBOS:
        print(f"\n{'=' * 70}\nCombo: {combo['name']}  (year_min={combo['year_min']}, "
              f"neighbours={len(combo['neighbours'])}, wind={len(combo['wind'])})\n{'=' * 70}")
        wave, X, sources = build_combo(combo)
        print(f"  window  {wave.index.min().date()} → {wave.index.max().date()}   "
              f"rows {len(wave):,}   features {X.shape[1]}")

        per_h: dict[int, dict[str, dict[str, float]]] = {}
        for h in HORIZONS_H:
            print(f"\n  --- h={h}h ---")
            t_cell = time.time()
            per_h[h] = run_cell(combo, h, wave, X, sources, log=True)
            ens = per_h[h]["ensemble"]
            pst = per_h[h]["persistence"]
            print(f"  [{time.time() - t_cell:5.1f}s] "
                  f"persistence RMSE {pst['RMSE']:.4f}  |  "
                  f"ensemble RMSE {ens['RMSE']:.4f}  Skill {ens['SkillVsBaseline']:+.4f}")
        results[combo["name"]] = per_h

    print(f"\nTotal sweep time: {time.time() - t0:.1f}s")

    # --- Summary table for the log -----------------------------------------
    print("\nSummary (ensemble RMSE / skill by horizon):")
    horizons = HORIZONS_H
    header = "horizon  " + "  ".join(f"{c['name']:>10}" for c in COMBOS)
    print(header)
    for h in horizons:
        skills = [f"{results[c['name']][h]['ensemble']['SkillVsBaseline']:+.4f}" for c in COMBOS]
        rmses  = [f"{results[c['name']][h]['ensemble']['RMSE']:.4f}"             for c in COMBOS]
        print(f"  h={h:>3}h  " + "  ".join(f"{r}" for r in rmses))
        print(f"   skill  " + "  ".join(f"{s}" for s in skills))

    plot_sweep(results, baselines)


if __name__ == "__main__":
    main()
