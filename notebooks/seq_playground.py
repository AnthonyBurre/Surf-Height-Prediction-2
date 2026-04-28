"""Sequence-model playground for +12h hsig_m.

Run:  ./.venv/bin/python notebooks/seq_playground.py

A single CONFIG dict at the top controls everything: data window, neighbour
buoys, wind stations, feature mode (raw vs engineered), model class, and all
hyperparameters. Edit it in place and re-run - no other code changes needed.
Each completed run appends to experiments.jsonl with a name derived from the
CONFIG, so back-to-back runs with different settings stay distinguishable.

Why this script exists
----------------------
The earlier LSTM (`mooloolaba_brisbane_lstm.py`) underperformed persistence
across every logged configuration (best skill: -55%). The two failure modes
called out in that script's header were:

  (a) Raw circular-encoded channels carry too little information — the model
      has to rediscover what the lag/momentum features encode explicitly.
  (b) CPU is slow enough that ramping epochs is painful (~25 min / 50 epochs
      at hidden=64).

This playground addresses both: switch ``feature_mode`` to ``"engineered"``
to give the model the same lag/rolling matrix the linear models use, and
flip ``subsample_steps`` to a small number for fast iteration before
committing to a full-length training run.

Starting points to try (paste into CONFIG)
------------------------------------------
1. Quick smoke (≈30s on CPU):
     model="gru", feature_mode="raw", epochs=3, subsample_steps=20000
2. Default sane run (≈8 min CPU):
     model="gru", feature_mode="raw", wind_stations=["mountain-creek"], epochs=15
3. Engineered features — should beat the raw-input LSTM:
     model="gru", feature_mode="engineered", wind_stations=["mountain-creek"], epochs=15
4. TCN, parallelisable conv-based:
     model="tcn", feature_mode="raw", wind_stations=["mountain-creek"], epochs=20
5. Throw everything in:
     model="gru", feature_mode="engineered",
     wind_stations=["mountain-creek", "deception-bay"],
     neighbours=["brisbane"], epochs=30, hidden=128

Tips
----
- ``device=None`` auto-detects (cuda > mps > cpu). On this Mac MPS is built
  but not currently available, so runs land on CPU.
- ``subsample_steps=N`` keeps only the LAST N training rows — fastest path
  to a usable signal during iteration. Set to None for full-length training.
- Persistence is computed on the same test window so the skill score is
  fair across CONFIG changes.
- Mean-imputation is mandatory before sequence models: with seq_len=48 and
  any column carrying a few % NaN, almost every window contains a NaN and
  training collapses (see header note in mooloolaba_brisbane_lstm.py).
"""

import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from sklearn.impute import SimpleImputer

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
    # --- data ---
    "year_max":      2024,        # cap wave history; 2024 = full wind overlap (both stations)
    "neighbours":    ["brisbane"],          # subset of: "brisbane", "caloundra", "goldcoast", "north-moreton-bay"
    "wind_stations": ["mountain-creek", "deception-bay"],  # any subset of: "mountain-creek", "deception-bay"; [] disables wind

    # --- features ---
    # "raw"        — circular-encoded raw channels + sin/cos time features
    #                (matches build_seq_features; what the previous LSTM used)
    # "engineered" — full lag + rolling + momentum matrix
    #                (matches build_mooloolaba_features; what Ridge uses)
    "feature_mode": "raw",

    # --- model ---
    "model":         "lstm",       # "lstm", "gru", "rnn", "tcn"

    # shared seq-model hyperparams
    "seq_len":       48,          # 48 × 30 min = 24 h of context
    "hidden":        96,
    "num_layers":    2,
    "epochs":        15,
    "batch_size":    512,
    "lr":            1e-3,
    "seed":          42,

    # TCN-only knobs (ignored for RNN/GRU/LSTM)
    "tcn_channels":     (64, 64, 64, 64),
    "tcn_kernel_size":  3,
    "tcn_dropout":      0.1,

    # --- run / logging ---
    "run_name":        "neuropt",     # log entries get suffixed with model+mode+wind
    "log_to_jsonl":    True,
    "verbose_train":   True,
    "device":          None,           # None = auto-detect (cuda > mps > cpu)
    "subsample_steps": None,           # e.g. 20000 to use only the last N train rows
}

# ---------------------------------------------------------------------------
# Internals — generally no need to touch below this line
# ---------------------------------------------------------------------------

_NEIGHBOUR_FILES = {
    "brisbane":          "brisbane_wave_data_2015-2025.csv",
    "caloundra":         "caloundra_wave_data_2013-2025.csv",
    "goldcoast":         "gold-coast_wave_data_2015-2025.csv",
    "north-moreton-bay": "north-moreton-bay_wave_data_2010-2025.csv",
}

_WIND_FILES = {
    "mountain-creek": "mountain-creek_wind_data_2015-2024.csv",
    "deception-bay":  "deception-bay_wind_data_2015-2024.csv",
}


def auto_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_wave(year_max: int) -> pd.DataFrame:
    df = fc.load_data()
    return df.loc[df.index.year <= year_max]


def load_neighbours(target_index: pd.DatetimeIndex, neighbours: list[str]) -> dict[str, pd.Series]:
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


def load_wind(target_index: pd.DatetimeIndex,
              stations: list[str]) -> pd.DataFrame | None:
    """Hourly wind for one or more stations, sin/cos-encoded, ffill'd to 30-min grid.

    Returns None if no stations are requested. Multi-station loads namespace
    each station's columns by slug (e.g. ``deception-bay_wind_speed_ms``).
    """
    if not stations:
        return None
    frames: list[pd.DataFrame] = []
    for s in stations:
        if s not in _WIND_FILES:
            raise ValueError(f"Unknown wind station {s!r}; supported: {list(_WIND_FILES)}")
        w = pd.read_csv(
            DATA_DIR / _WIND_FILES[s],
            parse_dates=["datetime_utc"], index_col="datetime_utc",
        )
        w = encode_circular(w, columns=["wind_dir_deg"])
        w = w.add_prefix(f"{s}_")
        frames.append(w.reindex(target_index, method="ffill"))
    return pd.concat(frames, axis=1)


def build_features(
    wave: pd.DataFrame,
    neighbour_series: dict[str, pd.Series],
    wind: pd.DataFrame | None,
    mode: str,
) -> pd.DataFrame:
    merged = wave.copy()
    neighbour_cols = []
    for name, series in neighbour_series.items():
        col = f"{name}_hsig_m"
        merged[col] = series
        neighbour_cols.append(col)

    if mode == "raw":
        # build_seq_features = encode_circular + add_time_features
        # neighbour cols ride through unchanged; wind cols appended below
        X = fc.build_seq_features(merged)
        if wind is not None:
            for col in wind.columns:
                X[col] = wind[col]
    elif mode == "engineered":
        mool_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
        X = fc.build_mooloolaba_features(mool_only)
        if neighbour_cols:
            X = fc.add_neighbour_features(X, merged, neighbour_cols)
        if wind is not None:
            X = fc.add_neighbour_features(X, wind, list(wind.columns))
    else:
        raise ValueError(f"feature_mode must be 'raw' or 'engineered', got {mode!r}")
    return X


def build_model(cfg: dict, device: str):
    common = dict(
        seq_len=cfg["seq_len"], hidden=cfg["hidden"], num_layers=cfg["num_layers"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        seed=cfg["seed"], device=device, verbose=cfg["verbose_train"],
    )
    name = cfg["model"].lower()
    if name == "lstm": return fc.LSTMForecaster(**common)
    if name == "gru":  return fc.GRUForecaster(**common)
    if name == "rnn":  return fc.SimpleRNNForecaster(**common)
    if name == "tcn":
        return fc.TCNForecaster(
            channels=cfg["tcn_channels"],
            kernel_size=cfg["tcn_kernel_size"],
            dropout=cfg["tcn_dropout"],
            **common,
        )
    raise ValueError(f"Unknown model {name!r}; choose lstm/gru/rnn/tcn")


def _wind_tag(stations: list[str]) -> str:
    return "+".join("".join(part[0] for part in s.split("-")) for s in stations)


def make_run_name(cfg: dict) -> str:
    parts = [cfg["run_name"], cfg["model"], cfg["feature_mode"]]
    if cfg["wind_stations"]:
        parts.append("wind-" + _wind_tag(cfg["wind_stations"]))
    if cfg["neighbours"]:
        parts.append("+".join(cfg["neighbours"]))
    return "_".join(parts)


def main() -> None:
    cfg = CONFIG
    device = auto_device(cfg["device"])
    print(f"device         : {device}")

    wave = load_wave(cfg["year_max"])
    neighbours = load_neighbours(wave.index, cfg["neighbours"])
    wind = load_wind(wave.index, cfg["wind_stations"])

    # Restrict to wind overlap when wind is in play, so every training row has wind.
    if wind is not None:
        valid = wind.dropna(how="all")
        start, end = valid.index.min(), valid.index.max()
        wave = wave.loc[start:end]
        neighbours = {k: v.loc[start:end] for k, v in neighbours.items()}
        wind = wind.loc[start:end]

    X = build_features(wave, neighbours, wind, cfg["feature_mode"])
    y = fc.make_target(wave)
    X_p = wave[["hsig_m"]]

    X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y)
    X_p_tr, X_p_te, _, _   = fc.chronological_split(X_p, y)

    if cfg["subsample_steps"] is not None:
        n = int(cfg["subsample_steps"])
        X_tr   = X_tr.iloc[-n:]
        y_tr   = y_tr.iloc[-n:]
        X_p_tr = X_p_tr.iloc[-n:]
        print(f"subsampling    : last {n:,} train rows")

    print(f"window         : {wave.index.min()} → {wave.index.max()}")
    print(f"feature_mode   : {cfg['feature_mode']}  ({X.shape[1]} features)")
    print(f"train rows     : {len(X_tr):,}    test rows: {len(X_te):,}")
    nan_pct = X.isna().mean().mul(100)
    worst = nan_pct.sort_values(ascending=False).head(3).round(2).to_dict()
    print(f"top NaN cols   : {worst}\n")

    imp = SimpleImputer(strategy="mean")
    X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_te_imp = pd.DataFrame(imp.transform(X_te),     columns=X_te.columns, index=X_te.index)

    persist = fc.evaluate(fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te)
    pp = persist.predictions
    print(f"persistence    : RMSE {persist.metrics['RMSE']:.4f}\n")

    model = build_model(cfg, device)
    name = make_run_name(cfg)
    print(f"=== {name}  (seq_len={cfg['seq_len']}, hidden={cfg['hidden']}, "
          f"layers={cfg['num_layers']}, epochs={cfg['epochs']}, lr={cfg['lr']}) ===")

    t0 = time.time()
    model.fit(X_tr_imp, y_tr)
    preds = model.predict(X_te_imp)
    elapsed = time.time() - t0
    print(f"\nfit + predict  : {elapsed/60:.2f} min")

    metrics = summarise(y_te.to_numpy(), preds, y_pred_baseline=pp)
    result = fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=model)

    print(f"\n{name}")
    print(f"  RMSE  {metrics['RMSE']:.4f}    Skill {metrics['SkillVsBaseline']:+.4f}")
    print(f"  MAE   {metrics['MAE']:.4f}    Bias  {metrics['Bias']:+.4f}")

    if cfg["log_to_jsonl"]:
        sources = ["mooloolaba"] + cfg["neighbours"] + cfg["wind_stations"]
        fc.log_run(
            result,
            data_sources=sources,
            train_index=X_tr.index,
            test_index=X_te.index,
            n_features=X.shape[1],
            extra={
                "feature_mode":     cfg["feature_mode"],
                "seq_len":          cfg["seq_len"],
                "hidden":           cfg["hidden"],
                "num_layers":       cfg["num_layers"],
                "epochs":           cfg["epochs"],
                "lr":               cfg["lr"],
                "batch_size":       cfg["batch_size"],
                "device":           device,
                "wind_stations":    cfg["wind_stations"],
                "subsample_steps":  cfg["subsample_steps"],
                "elapsed_min":      round(elapsed / 60, 2),
                "imputation":       "mean",
            },
        )

    log = fc.read_log()
    recent = log[log["name"].str.startswith(cfg["run_name"])].sort_values("timestamp").tail(8)
    if len(recent) > 1:
        print("\nrecent playground runs (most recent last):")
        for _, row in recent.iterrows():
            m = row["metrics"]
            extra = row.get("extra") or {}
            tag = f"{extra.get('feature_mode','?')}|{extra.get('epochs','?')}ep|h{extra.get('hidden','?')}"
            print(f"  {row['name']:55s}  {tag:24s}  RMSE {m['RMSE']:.4f}  Skill {m.get('SkillVsBaseline',0):+.4f}")


if __name__ == "__main__":
    main()
