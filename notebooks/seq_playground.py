"""Sequence-model playground for +12h hsig_m.

Run:  ./.venv/bin/python notebooks/seq_playground.py

A single CONFIG dict at the top controls everything: data window, neighbour
buoys, wind stations, feature mode (raw vs engineered), model class, and all
hyperparameters. Edit it in place and re-run - no other code changes needed.
Each completed run appends to experiments.jsonl with a name derived from the
CONFIG, so back-to-back runs with different settings stay distinguishable.

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

import pandas as pd

import forecast as fc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

# ---------------------------------------------------------------------------
# CONFIG — edit me
# ---------------------------------------------------------------------------

CONFIG: dict = {
    # --- data ---
    "primary_buoy":  "mooloolaba",  # any key in qld_ckan.wave.constants.BUOYS; download with `python -m qld_ckan wave --buoy NAME`
    "year_max":      2024,        # cap wave history; 2024 = full wind overlap (both stations)
    "neighbours":    ["brisbane", "gold-coast", "north-moreton-bay"],          # subset of: "brisbane", "caloundra", "gold-coast", "north-moreton-bay"
    "wind_stations": ["mountain-creek", "deception-bay"],  # any subset of: "mountain-creek", "deception-bay"; [] disables wind

    # --- features ---
    # "raw"        — circular-encoded raw channels + sin/cos time features
    #                (matches build_seq_features; what the previous LSTM used)
    # "engineered" — full lag + rolling + momentum matrix
    #                (matches build_buoy_features; what Ridge uses)
    # "raw" beats "engineered" for every seq model — see seq_sweep.py.
    "feature_mode": "raw",

    # --- model ---
    "model":         "rnn",       # "lstm", "gru", "rnn", "tcn"

    # shared seq-model hyperparams.
    # Defaults below are the best RNN config from notebooks/seq_sweep.py
    # (RMSE 0.2324, skill +0.234 vs persistence). The key lesson from the
    # sweep: these models overfit the persistence residual fast — epochs=2-3
    # beats epochs=5 every time, and "raw" features beat "engineered".
    # Best config per model class (all on feature_mode="raw", lr=1e-3):
    #   rnn  : seq_len=48 hidden=128 num_layers=2 epochs=3  → skill +0.234
    #   gru  : seq_len=48 hidden=64  num_layers=1 epochs=2  → skill +0.230
    #   lstm : seq_len=48 hidden=64  num_layers=1 epochs=3  → skill +0.157
    "seq_len":       48,          # 48 × 30 min = 24 h of context
    "hidden":        128,
    "num_layers":    2,
    "epochs":        3,
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


def load_wave(buoy: str, year_max: int) -> pd.DataFrame:
    df = fc.load_data(buoy=buoy)
    return df.loc[df.index.year <= year_max]


def build_features(
    merged: pd.DataFrame,
    neighbour_cols: list[str],
    wind: pd.DataFrame | None,
    mode: str,
) -> pd.DataFrame:
    if mode == "raw":
        # build_seq_features = encode_circular + add_time_features
        # neighbour cols ride through unchanged; wind cols appended below
        X = fc.build_seq_features(merged)
        if wind is not None:
            for col in wind.columns:
                X[col] = wind[col]
    elif mode == "engineered":
        primary_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
        X = fc.build_buoy_features(primary_only)
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


def main() -> None:
    cfg = CONFIG
    device = fc.auto_device(cfg["device"])
    print(f"device         : {device}")

    wave = load_wave(cfg["primary_buoy"], cfg["year_max"])
    neighbours = fc.load_neighbours(wave.index, cfg["neighbours"])
    wind = fc.load_wind(wave.index, cfg["wind_stations"])

    # Restrict to wind overlap when wind is in play, so every training row has wind.
    wave, neighbours, wind = fc.restrict_to_overlap(wave, neighbours, wind)

    merged, neighbour_cols, _ = fc.assemble_inputs(wave, neighbours, wind)
    X = build_features(merged, neighbour_cols, wind, cfg["feature_mode"])
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

    X_tr_imp, X_te_imp = fc.mean_impute(X_tr, X_te)

    persist = fc.evaluate(fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te)
    pp = persist.predictions
    print(f"persistence    : RMSE {persist.metrics['RMSE']:.4f}\n")

    model = build_model(cfg, device)
    name = fc.compose_run_name(
        cfg["run_name"],
        model=cfg["model"],
        feature_mode=cfg["feature_mode"],
        wind_stations=cfg["wind_stations"],
        neighbours=cfg["neighbours"],
    )
    print(f"=== {name}  (seq_len={cfg['seq_len']}, hidden={cfg['hidden']}, "
          f"layers={cfg['num_layers']}, epochs={cfg['epochs']}, lr={cfg['lr']}) ===")

    t0 = time.time()
    model.fit(X_tr_imp, y_tr)
    preds = model.predict(X_te_imp)
    elapsed = time.time() - t0
    print(f"\nfit + predict  : {elapsed/60:.2f} min")

    metrics = fc.summarise(y_te.to_numpy(), preds, y_pred_baseline=pp)
    result = fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=model)

    print(f"\n{name}")
    print(f"  RMSE  {metrics['RMSE']:.4f}    Skill {metrics['SkillVsBaseline']:+.4f}")
    print(f"  MAE   {metrics['MAE']:.4f}    Bias  {metrics['Bias']:+.4f}")

    if cfg["log_to_jsonl"]:
        sources = [cfg["primary_buoy"]] + cfg["neighbours"] + cfg["wind_stations"]
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

    recent = fc.recent_runs(cfg["run_name"], n=8)
    if len(recent) > 1:
        print("\nrecent playground runs (most recent last):")
        for _, row in recent.iterrows():
            m = row["metrics"]
            extra = row.get("extra") or {}
            tag = f"{extra.get('feature_mode','?')}|{extra.get('epochs','?')}ep|h{extra.get('hidden','?')}"
            print(f"  {row['name']:55s}  {tag:24s}  RMSE {m['RMSE']:.4f}  Skill {m.get('SkillVsBaseline',0):+.4f}")


if __name__ == "__main__":
    main()
