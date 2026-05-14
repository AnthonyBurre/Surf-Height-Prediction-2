"""Hyperparameter sweep for the RNN / GRU / LSTM sequence forecasters.

Run:  ./.venv/bin/python notebooks/seq_sweep.py

Reuses seq_playground's data-loading / feature-building helpers so the
sweep sees exactly the same inputs a single playground run would. Builds
the data + persistence baseline once, then loops a small grid over each
model class. Every run is logged to experiments.jsonl under the
``seqsweep`` prefix so results stay distinguishable from manual runs.

The grid is deliberately small and low-epoch: history (experiments.jsonl)
shows these models overfit the persistence residual quickly, so the goal
is to find the few low-capacity / low-epoch corners that actually beat
persistence rather than to train hard.
"""

import time
import warnings

import pandas as pd

import forecast as fc
from seq_playground import build_features, load_wave

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

# --- fixed data config (mirrors seq_playground CONFIG) ---
PRIMARY_BUOY = "mooloolaba"
YEAR_MAX = 2024
NEIGHBOURS = ["brisbane", "caloundra", "gold-coast", "north-moreton-bay"]
WIND_STATIONS = ["mountain-creek", "deception-bay"]
FEATURE_MODE = "raw"  # raw = 24 circular-encoded channels; fastest + best historically
RUN_PREFIX = "seqsweep"

# --- the grid ---
MODELS = ["rnn", "gru", "lstm", "tcn"]
GRID = [
    # (seq_len, hidden, num_layers, epochs)
    (48, 32, 1, 3),
    (48, 64, 1, 3),
    (48, 64, 2, 3),
    (48, 128, 1, 3),
    (48, 128, 2, 3),
    (24, 64, 1, 3),
]
# extra epoch probe on the smallest-capacity corner (overfitting check)
EPOCH_PROBE = [(48, 64, 1, 2), (48, 64, 1, 5)]

BATCH_SIZE = 512
LR = 1e-3
SEED = 42


def build_data():
    """Build X / y / persistence once; return everything the sweep needs."""
    wave = load_wave(PRIMARY_BUOY, YEAR_MAX)
    neighbours = fc.load_neighbours(wave.index, NEIGHBOURS)
    wind = fc.load_wind(wave.index, WIND_STATIONS)
    wave, neighbours, wind = fc.restrict_to_overlap(wave, neighbours, wind)

    merged, neighbour_cols, _ = fc.assemble_inputs(wave, neighbours, wind)
    X = build_features(merged, neighbour_cols, wind, FEATURE_MODE)
    y = fc.make_target(wave)
    X_p = wave[["hsig_m"]]

    X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y)
    X_p_tr, X_p_te, _, _ = fc.chronological_split(X_p, y)
    X_tr_imp, X_te_imp = fc.mean_impute(X_tr, X_te)

    persist = fc.evaluate(fc.PersistenceForecaster(), X_p_tr, y_tr, X_p_te, y_te)
    return {
        "X_tr": X_tr_imp, "X_te": X_te_imp, "y_tr": y_tr, "y_te": y_te,
        "train_index": X_tr.index, "test_index": X_te.index,
        "n_features": X.shape[1],
        "persist_preds": persist.predictions,
        "persist_rmse": persist.metrics["RMSE"],
    }


def make_model(model: str, seq_len: int, hidden: int, num_layers: int, epochs: int):
    common = dict(
        seq_len=seq_len, epochs=epochs,
        batch_size=BATCH_SIZE, lr=LR, seed=SEED, device="cpu", verbose=False,
    )
    if model == "tcn":
        # TCN has no hidden/num_layers; reuse those grid axes as the
        # width/depth of a uniform dilated-conv stack.
        return fc.TCNForecaster(channels=(hidden,) * num_layers, **common)
    return {
        "rnn": fc.SimpleRNNForecaster,
        "gru": fc.GRUForecaster,
        "lstm": fc.LSTMForecaster,
    }[model](hidden=hidden, num_layers=num_layers, **common)


def run_one(d: dict, model: str, seq_len: int, hidden: int, num_layers: int,
            epochs: int) -> dict:
    name = f"{RUN_PREFIX}_{model}_{FEATURE_MODE}_sl{seq_len}_h{hidden}_L{num_layers}_ep{epochs}"
    m = make_model(model, seq_len, hidden, num_layers, epochs)
    t0 = time.time()
    m.fit(d["X_tr"], d["y_tr"])
    preds = m.predict(d["X_te"])
    elapsed = time.time() - t0

    metrics = fc.summarise(
        d["y_te"].to_numpy(), preds, y_pred_baseline=d["persist_preds"]
    )
    result = fc.EvaluationResult(name=name, metrics=metrics, predictions=preds, model=m)
    fc.log_run(
        result,
        data_sources=[PRIMARY_BUOY] + NEIGHBOURS + WIND_STATIONS,
        train_index=d["train_index"],
        test_index=d["test_index"],
        n_features=d["n_features"],
        extra={
            "feature_mode": FEATURE_MODE, "seq_len": seq_len, "hidden": hidden,
            "num_layers": num_layers, "epochs": epochs, "lr": LR,
            "batch_size": BATCH_SIZE, "device": "cpu", "imputation": "mean",
            "elapsed_min": round(elapsed / 60, 2), "sweep": True,
        },
    )
    row = {
        "model": model, "seq_len": seq_len, "hidden": hidden,
        "num_layers": num_layers, "epochs": epochs,
        "RMSE": metrics["RMSE"], "Skill": metrics["SkillVsBaseline"],
        "MAE": metrics["MAE"], "Bias": metrics["Bias"], "secs": round(elapsed, 1),
    }
    flag = "  <-- beats persistence" if metrics["SkillVsBaseline"] > 0 else ""
    print(f"  {name:48s}  RMSE {metrics['RMSE']:.4f}  "
          f"Skill {metrics['SkillVsBaseline']:+.4f}  ({elapsed:5.1f}s){flag}", flush=True)
    return row


def main() -> None:
    print(f"building data ({FEATURE_MODE} mode)...", flush=True)
    d = build_data()
    print(f"persistence RMSE: {d['persist_rmse']:.4f}   "
          f"(features={d['n_features']}, train={len(d['y_tr']):,}, "
          f"test={len(d['y_te']):,})\n", flush=True)

    rows: list[dict] = []
    for model in MODELS:
        print(f"=== {model.upper()} ===", flush=True)
        for seq_len, hidden, num_layers, epochs in GRID:
            rows.append(run_one(d, model, seq_len, hidden, num_layers, epochs))
        for seq_len, hidden, num_layers, epochs in EPOCH_PROBE:
            rows.append(run_one(d, model, seq_len, hidden, num_layers, epochs))
        print(flush=True)

    summary = pd.DataFrame(rows).sort_values("Skill", ascending=False)
    print("=== full sweep, best skill first ===")
    print(summary.to_string(index=False))

    print("\n=== best config per model ===")
    for model in MODELS:
        best = summary[summary["model"] == model].iloc[0]
        beat = "BEATS baseline" if best["Skill"] > 0 else "below baseline"
        print(f"  {model:5s}: sl{int(best['seq_len'])} h{int(best['hidden'])} "
              f"L{int(best['num_layers'])} ep{int(best['epochs'])}  "
              f"RMSE {best['RMSE']:.4f}  Skill {best['Skill']:+.4f}  ({beat})")


if __name__ == "__main__":
    main()
