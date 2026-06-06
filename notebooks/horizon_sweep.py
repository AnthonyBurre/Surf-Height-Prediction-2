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
Lasso, HGB-on-persistence-residual, a nanmean ensemble of the three, plus
the four sequence forecasters (RNN/GRU/LSTM/TCN) using the best hyperparams
picked from the narrow + wide sequence sweeps. Sequence models only run on
the `baseline` and `wide` combos — they need a full circular-encoded input
frame and the seq sweep only fitted those two feature sets.

All runs share the same pinned test-window cutoff (2023-01-01 AEST) so skill
scores at any one horizon are directly comparable across combos. Persistence
is computed per-horizon (it varies a lot — the autocorrelation collapses).

Saves:
  horizon_sweep.png — single panel: RMSE vs horizon for just the per-horizon
                       winners. For each horizon we identify the (combo, arch)
                       with the lowest RMSE; each distinct winning pair gets a
                       full-trajectory line across all 6 horizons, with a star
                       marking the horizon(s) it wins. Persistence is drawn as
                       a black dashed reference line.
  experiments.jsonl entries under names 'hsweep_<combo>_h<H>h_<model>' and
                       'hsweep_seq_<combo>_h<H>h_<arch>'.

Linear cells are not relogged on rerun (log=False) — the first run already
laid down 120 hsweep entries and the in-memory metrics are what the plot
uses. Sequence cells log every time.

Iterate on the chart without paying the 90-min sweep cost:
    ./.venv/bin/python notebooks/horizon_sweep.py --plot-only
which reads results from experiments.jsonl and just re-renders the figure.
"""
import sys
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

# Notebooks share helpers via sys.path-relative imports.
sys.path.insert(0, str(Path(__file__).parent))
from seq_playground import build_features as build_seq_features_frame  # noqa: E402

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
LINEAR_ARCHITECTURES = ["ridge", "lasso", "hgb"]
SEQ_ARCHITECTURES    = ["rnn", "gru", "lstm", "tcn"]
ARCHITECTURES        = LINEAR_ARCHITECTURES + SEQ_ARCHITECTURES

# Combos that have a fitted sequence-sweep config. Solo / tweed_mc don't —
# they'd need their own seq sweep first (probably not interesting; the
# sequence models depend on neighbour breadth to win).
SEQ_COMBOS = ["baseline", "wide"]

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

# Best (combo, arch) sequence-model config from the seq sweeps. Each value
# is the kwargs dict that build_seq_model() unpacks into the matching
# Forecaster constructor. Picked by lowest RMSE within each (set, arch)
# group of the seqsweep_<set>_<arch>_* rows in experiments.jsonl.
SEQ_CONFIGS: dict[tuple[str, str], dict] = {
    # --- baseline (narrow) ---
    ("baseline", "rnn"):  {"seq_len": 48, "hidden": 128, "num_layers": 2,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.0},
    ("baseline", "gru"):  {"seq_len": 48, "hidden":  64, "num_layers": 1,
                            "epochs": 2, "weight_decay": 0.0,  "rnn_dropout": 0.0},
    ("baseline", "lstm"): {"seq_len": 48, "hidden": 128, "num_layers": 1,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.0},
    ("baseline", "tcn"):  {"seq_len": 48, "hidden": 128, "num_layers": 2,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.2},
    # --- wide ---
    ("wide",     "rnn"):  {"seq_len": 48, "hidden": 256, "num_layers": 2,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.1},
    ("wide",     "gru"):  {"seq_len": 48, "hidden":  64, "num_layers": 1,
                            "epochs": 2, "weight_decay": 0.0,  "rnn_dropout": 0.0},
    ("wide",     "lstm"): {"seq_len": 48, "hidden": 128, "num_layers": 2,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.1},
    ("wide",     "tcn"):  {"seq_len": 48, "hidden": 128, "num_layers": 4,
                            "epochs": 3, "weight_decay": 1e-4, "rnn_dropout": 0.2},
}

SEQ_BATCH_SIZE = 512
SEQ_LR = 1e-3
SEQ_SEED = 42
SEQ_SCALER = "robust"


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


def build_seq_combo(combo: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Same window as build_combo but emits the circular-encoded seq frame.

    Sequence forecasters window their own input over time and expect raw
    channels + sin/cos direction columns, NOT the lag/rolling feature matrix
    used by the linear/tree models. Mirrors seq_playground's data path so
    the runs here are directly comparable to the seq sweep tables.
    """
    wave = fc.restrict_to_years(
        fc.load_data(buoy="mooloolaba"), combo["year_min"], combo["year_max"],
    )
    neighbours = fc.load_neighbours(wave.index, combo["neighbours"])
    wind = fc.load_wind(wave.index, combo["wind"])
    wave, neighbours, wind = fc.restrict_to_overlap(wave, neighbours, wind)

    merged, neighbour_cols, _ = fc.assemble_inputs(wave, neighbours, wind)
    X = build_seq_features_frame(merged, neighbour_cols, wind, "raw")
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
    horizon_steps = fc.hours_to_steps(horizon_h)
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
    horizon_steps = fc.hours_to_steps(horizon_h)
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
# Sequence cell — fit one (arch, combo, horizon)
# ---------------------------------------------------------------------------

def _build_seq_model(arch: str, cfg: dict):
    common = dict(
        seq_len=cfg["seq_len"], epochs=cfg["epochs"],
        batch_size=SEQ_BATCH_SIZE, lr=SEQ_LR, seed=SEQ_SEED,
        device=fc.auto_device(), verbose=False, scaler=SEQ_SCALER,
        weight_decay=cfg["weight_decay"],
    )
    if arch == "tcn":
        return fc.TCNForecaster(
            channels=(cfg["hidden"],) * cfg["num_layers"],
            dropout=cfg["rnn_dropout"], **common,
        )
    cls = {"rnn": fc.SimpleRNNForecaster,
           "gru": fc.GRUForecaster,
           "lstm": fc.LSTMForecaster}[arch]
    return cls(hidden=cfg["hidden"], num_layers=cfg["num_layers"],
               rnn_dropout=cfg["rnn_dropout"], **common)


def run_seq_cell(
    combo: dict,
    horizon_h: int,
    arch: str,
    cfg: dict,
    wave: pd.DataFrame,
    X_seq: pd.DataFrame,
    sources: list[str],
    *,
    log: bool,
) -> dict[str, float]:
    """Fit one sequence forecaster at one horizon; return metrics dict.

    Persistence at this (combo, horizon) is computed locally so skill is
    against the same baseline as the linear runs. Re-uses the pinned
    TEST_START split exactly like run_cell so seq numbers slot into the
    same plot rows.
    """
    horizon_steps = fc.hours_to_steps(horizon_h)
    y = fc.make_target(wave, horizon_steps=horizon_steps)

    ts = pd.Timestamp(TEST_START).tz_localize(fc.SOURCE_TZ)
    pos   = _pinned_split(X_seq.index, ts)
    pos_w = _pinned_split(wave.index,   ts)
    X_tr, X_te = X_seq.iloc[:pos], X_seq.iloc[pos:]
    y_tr, y_te = y.iloc[:pos],     y.iloc[pos:]
    X_p_tr = wave[["hsig_m"]].iloc[:pos_w]
    X_p_te = wave[["hsig_m"]].iloc[pos_w:]

    X_tr_imp, X_te_imp = fc.mean_impute(X_tr, X_te)
    persist = fc.evaluate(
        fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te,
        name=f"persistence_h{horizon_h}h_{combo['name']}",
    )
    pp = persist.predictions

    model = _build_seq_model(arch, cfg)
    t0 = time.time()
    model.fit(X_tr_imp, y_tr)
    preds = model.predict(X_te_imp)
    elapsed = time.time() - t0
    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=pp)

    name = f"hsweep_seq_{combo['name']}_h{horizon_h}h_{arch}"
    if log:
        fc.log_run(
            fc.EvaluationResult(name=name, metrics=metrics,
                                predictions=preds, model=model),
            data_sources=sources,
            train_index=X_tr.index, test_index=X_te.index,
            n_features=X_seq.shape[1],
            extra={
                "combo": combo["name"], "horizon_h": horizon_h,
                "horizon_steps": horizon_steps, "feature_mode": "raw",
                "seq_len": cfg["seq_len"], "hidden": cfg["hidden"],
                "num_layers": cfg["num_layers"], "epochs": cfg["epochs"],
                "lr": SEQ_LR, "batch_size": SEQ_BATCH_SIZE,
                "scaler": SEQ_SCALER, "device": fc.auto_device(),
                "imputation": "mean",
                "weight_decay": cfg["weight_decay"],
                "rnn_dropout": cfg["rnn_dropout"],
                "elapsed_min": round(elapsed / 60, 2),
            },
        )
    return metrics


# ---------------------------------------------------------------------------
# Plot — chart code lives in viz.results; this notebook just reshapes hsweep
# rows from experiments.jsonl into the long-format DataFrame the viz helper
# expects, and adds the persistence baseline row.
# ---------------------------------------------------------------------------

import re

from viz.results import plot_horizon_winners

_NAME_PAT_SEQ = re.compile(r"^hsweep_seq_(\w+)_h(\d+)h_(\w+)$")
_NAME_PAT_LIN = re.compile(r"^hsweep_(\w+)_h(\d+)h_(\w+)$")
_NAME_PAT_TWEED_SEQ = re.compile(r"^seqsweep_tweed_mc_(rnn|gru|lstm|tcn)_.*_h(\d+)h$")


def _runs_dataframe() -> pd.DataFrame:
    """Long-format ``hsweep_*`` runs: one row per (combo, arch, horizon).

    Pulls all matching rows out of ``experiments.jsonl`` via ``find_runs``,
    parses ``combo`` / ``horizon`` / ``arch`` out of the run name, then
    keeps only the most recent row per (combo, h, arch) so reruns
    naturally supersede older entries. Returns columns:
    ``label``, ``horizon_h``, ``RMSE``, plus ``combo`` / ``arch`` for
    debugging or further filtering.
    """
    df = fc.find_runs(name_prefix="hsweep_")
    rows: list[dict] = []
    for _, r in df.iterrows():
        m = _NAME_PAT_SEQ.match(r["name"]) or _NAME_PAT_LIN.match(r["name"])
        if not m:
            continue
        combo, h, arch = m.group(1), int(m.group(2)), m.group(3)
        rows.append({
            "combo": combo, "arch": arch, "horizon_h": h,
            "label": f"{combo} / {arch}" if arch != "persistence" else "persistence",
            "RMSE": r["metrics"]["RMSE"],
            "ts": r["timestamp"],
        })
    out = (
        pd.DataFrame(rows)
        .sort_values("ts")
        .drop_duplicates(["combo", "horizon_h", "arch"], keep="last")
        .drop(columns=["ts"])
        .reset_index(drop=True)
    )
    return out


def _tweed_mc_seq_rows() -> pd.DataFrame:
    """Best-config-per-(arch, horizon) rows from the tweed_mc seq sweep.

    Pulls every ``seqsweep_tweed_mc_*`` entry, groups by (arch, horizon),
    keeps the lowest-RMSE config per group, and emits long-format rows
    labelled ``tweed_mc / <arch>`` so they slot into the chart alongside
    the linear ``hsweep_`` and seq ``hsweep_seq_`` rows.
    """
    df = fc.find_runs(name_prefix="seqsweep_tweed_mc_")
    rows: list[dict] = []
    for _, r in df.iterrows():
        m = _NAME_PAT_TWEED_SEQ.match(r["name"])
        if not m:
            continue
        rows.append({
            "arch": m.group(1), "horizon_h": int(m.group(2)),
            "RMSE": r["metrics"]["RMSE"],
        })
    if not rows:
        return pd.DataFrame(columns=["combo", "arch", "horizon_h", "label", "RMSE"])
    best = (
        pd.DataFrame(rows)
        .sort_values("RMSE")
        .drop_duplicates(["arch", "horizon_h"], keep="first")
    )
    best["combo"] = "tweed_mc"
    best["label"] = "tweed_mc / " + best["arch"]
    return best[["combo", "arch", "horizon_h", "label", "RMSE"]].reset_index(drop=True)


def _climatology_rows(horizons: list[int]) -> pd.DataFrame:
    """Compute ClimatologyHour RMSE per horizon on the pinned test window.

    Cheap (Mooloolaba-only, no model fit) so we recompute on every plot
    call rather than relying on log entries that may have drifted. Returns
    rows in the same long format as ``_runs_dataframe`` so they can be
    concatenated directly.
    """
    wave_full = fc.restrict_to_years(fc.load_data(buoy="mooloolaba"), None, 2024)
    rows = []
    for h in horizons:
        rmse = compute_baselines(wave_full, h)["climatology_hour"]["RMSE"]
        rows.append({"combo": None, "arch": "climatology_hour", "horizon_h": h,
                     "label": "climatology hour", "RMSE": float(rmse)})
    return pd.DataFrame(rows)


def save_horizon_winners_chart(runs: pd.DataFrame) -> None:
    """Render the per-horizon-winners chart to ``figures/horizon_sweep.png``.

    Single RMSE-vs-horizon panel focused on the architecture × dataset
    story: for each horizon the lowest-RMSE (non-ensemble) model wins,
    every distinct winner gets its full trajectory across all horizons,
    and stars mark its winning horizon(s). The two no-model baselines
    (persistence + climatology-hour) are collapsed via viz's
    ``collapse_baselines`` and drawn as a faint gray backdrop — the
    crossover narrative lives in the README, not the chart.
    """
    horizons = sorted(runs["horizon_h"].unique())

    # Drop ensembles — they muddy the per-model comparison; the user wants
    # to see which individual model wins at each horizon.
    runs = runs[runs["arch"] != "ensemble"].copy()
    # Append tweed_mc seq best-config-per-arch + climatology baseline rows.
    runs = pd.concat(
        [runs, _tweed_mc_seq_rows(), _climatology_rows(horizons)],
        ignore_index=True,
    )

    print("\n=== Per-horizon winners (ensembles excluded) ===")
    contenders = runs[~runs["label"].isin(["persistence", "climatology hour"])]
    for h in horizons:
        sub = contenders[contenders["horizon_h"] == h]
        if sub.empty:
            continue
        win = sub.loc[sub["RMSE"].idxmin()]
        print(f"  h={h:>3}h  {win['combo']:>10}/{win['arch']:<10}  RMSE {win['RMSE']:.4f}")

    # Baselines collapsed to one backdrop line — light gray, thinner, no
    # marker — so the eye lands on the model trajectories first.
    baseline_style = {
        "persistence":      {"linestyle": "--", "color": "#999999",
                             "alpha": 0.45, "linewidth": 1.2, "marker": ""},
        "climatology hour": {"linestyle": "--", "color": "#999999",
                             "alpha": 0.45, "linewidth": 1.2, "marker": ""},
    }
    title = (
        "Mooloolaba significant wave height — best model per forecast horizon\n"
        "pinned 2023-01-01 → 2024-12-31 test window  ·  ★ marks the lowest-RMSE model at each horizon"
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plot_horizon_winners(
        runs, horizon_col="horizon_h", metric_col="RMSE", label_col="label",
        baseline_label=baseline_style, collapse_baselines=True,
        title=title, ax=ax,
    )
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("RMSE (m)")
    fig.tight_layout()
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
            # log=False: 120 linear hsweep entries are already in
            # experiments.jsonl from the first run; rerunning is fast and
            # gives us fresh metrics, but logging again would duplicate.
            per_h[h] = run_cell(combo, h, wave, X, sources, log=False)
            ens = per_h[h]["ensemble"]
            pst = per_h[h]["persistence"]
            print(f"  [{time.time() - t_cell:5.1f}s] "
                  f"persistence RMSE {pst['RMSE']:.4f}  |  "
                  f"ensemble RMSE {ens['RMSE']:.4f}  Skill {ens['SkillVsBaseline']:+.4f}")
        results[combo["name"]] = per_h

    print(f"\nLinear sweep time: {time.time() - t0:.1f}s")

    # --- Sequence models (only for combos with a fitted seq config) ------
    t_seq = time.time()
    seq_combos = [c for c in COMBOS if c["name"] in SEQ_COMBOS]
    for combo in seq_combos:
        print(f"\n{'=' * 70}\nSeq combo: {combo['name']}\n{'=' * 70}")
        wave, X_seq, sources = build_seq_combo(combo)
        print(f"  seq window  {wave.index.min().date()} → {wave.index.max().date()}   "
              f"rows {len(wave):,}   features {X_seq.shape[1]}")
        for arch in SEQ_ARCHITECTURES:
            cfg = SEQ_CONFIGS[(combo["name"], arch)]
            print(f"\n  === {arch.upper()}  h{cfg['hidden']} L{cfg['num_layers']} "
                  f"ep{cfg['epochs']} wd{cfg['weight_decay']:g} do{cfg['rnn_dropout']:g} ===")
            for h in HORIZONS_H:
                t_cell = time.time()
                metrics = run_seq_cell(combo, h, arch, cfg, wave, X_seq, sources, log=True)
                results[combo["name"]][h][arch] = metrics
                print(f"    h={h:>3}h  [{time.time() - t_cell:6.1f}s]  "
                      f"RMSE {metrics['RMSE']:.4f}  Skill {metrics['SkillVsBaseline']:+.4f}")
    print(f"\nSeq sweep time: {time.time() - t_seq:.1f}s")
    print(f"Total sweep time: {time.time() - t0:.1f}s")

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

    save_horizon_winners_chart(_runs_dataframe())


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        # Replot from experiments.jsonl without re-running the 90-min sweep.
        save_horizon_winners_chart(_runs_dataframe())
    else:
        main()
