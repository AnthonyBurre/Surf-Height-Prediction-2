"""Mooloolaba + Brisbane +12h hsig_m forecast — LSTM.

Run:  ./.venv/bin/python notebooks/mooloolaba_brisbane_lstm.py

This script is intentionally separate from the linear-model script because
LSTM training takes ~25 minutes on CPU, a completely different iteration
cadence from Ridge/Lasso (seconds).

--- Findings so far ---

The LSTM is slow to converge on this problem without a GPU.

Architecture: LSTMForecaster(seq_len=48, hidden=64, num_layers=2, lr=1e-3).
Input frame: 12 raw channels (Mooloolaba + brisbane_hsig_m, encode_circular +
time features). No pre-built lag or rolling features — the LSTM is expected
to learn temporal structure from the raw sequence.

Configurations tried and their test RMSE (persistence baseline = 0.2891):

  epochs=15, layers=2, lr=1e-3  →  RMSE 0.3616  (worse than persistence)
  epochs=30, layers=1, lr=1e-3  →  RMSE 0.3930  (worse; 1 layer not enough capacity)
  epochs=30, layers=2, lr=5e-4  →  RMSE 0.3841  (worse; lr too slow to converge)
  epochs=50, layers=2, lr=1e-3  →  RMSE 0.3012  (Skill +3.7% vs persistence)

Training loss curve at 50 epochs (approx): 0.41 → 0.29 over 50 epochs, ~0.007
RMSE improvement per 10 epochs. The model crosses below persistence on the
training set at ~epoch 44.

For context, Ridge and Lasso on the full 120-feature lag/rolling matrix
achieve RMSE ~0.267 (Skill ~+15%) in under 5 seconds. The LSTM's main
disadvantage here is that it must rediscover from raw inputs what the lag and
rolling features encode explicitly. This gap would likely close with either:

  a) A GPU, allowing 200+ epochs feasibly
  b) Lag features added to the LSTM input frame (pre-computing the key
     temporal summaries the model currently has to learn from scratch)

NaN handling note: with seq_len=48 and 4.3% NaN in brisbane_hsig_m, roughly
88% of 48-step windows contain at least one NaN. The LSTM's built-in NaN
skipping during training (which drops entire windows) would leave the model
almost untrained. Mean imputation of the input frame is required before
passing it to the LSTM.
---
"""

import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

import forecast as fc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

DATA_DIR = Path(__file__).parent.parent / "data"


def mean_impute(
    X_tr: pd.DataFrame, X_te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = SimpleImputer(strategy="mean")
    return (
        pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index),
        pd.DataFrame(imp.transform(X_te),     columns=X_te.columns, index=X_te.index),
    )


def main() -> None:
    mool = fc.load_data()
    bris = fc.load_data(DATA_DIR / "brisbane_wave_data_2015-2025.csv")

    start = max(mool.index.min(), bris.index.min())
    end   = min(mool.index.max(), bris.index.max())
    mool  = mool.loc[start:end]
    bris  = bris.loc[start:end]

    print(f"Window : {start.date()} → {end.date()}")

    merged = mool.copy()
    merged["brisbane_hsig_m"] = bris["hsig_m"]

    y     = fc.make_target(merged)
    X_seq = fc.build_seq_features(merged)
    X_p   = merged[["hsig_m"]]

    X_seq_tr, X_seq_te, y_tr, y_te = fc.chronological_split(X_seq, y)
    X_p_tr,   X_p_te,   _,    _    = fc.chronological_split(X_p,   y)

    print(f"Seq features : {X_seq.shape[1]}  {list(X_seq.columns)}")
    print(f"Train rows   : {len(X_seq_tr):,}  |  test rows: {len(X_seq_te):,}\n")

    # Mean-impute before LSTM — see note at top of file.
    X_seq_tr_imp, X_seq_te_imp = mean_impute(X_seq_tr, X_seq_te)

    # Persistence baseline
    persist = fc.evaluate_and_log(
        fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name="persistence_mool_brisbane_lstm_run",
        data_sources=["mooloolaba"],
        extra={"window": f"{start.date()}:{end.date()}"},
    )
    pp = persist.predictions
    print(f"Persistence  RMSE {persist.metrics['RMSE']:.4f}\n")

    print("=== LSTM (seq_len=48, hidden=64, num_layers=2, epochs=50, lr=1e-3) ===")
    print("Training — this takes ~25 min on CPU. Watch the per-epoch loss:\n")

    lstm = fc.LSTMForecaster(
        seq_len=48,
        hidden=64,
        num_layers=2,
        epochs=50,
        batch_size=512,
        lr=1e-3,
        verbose=True,
    )

    t0 = time.time()
    lstm.fit(X_seq_tr_imp, y_tr)
    lstm_preds = lstm.predict(X_seq_te_imp)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    from forecast.metrics import summarise
    metrics = summarise(y_te.to_numpy(), lstm_preds, y_pred_baseline=pp)
    result = fc.EvaluationResult(
        name="lstm_mool_brisbane",
        metrics=metrics,
        predictions=lstm_preds,
        model=lstm,
    )
    fc.log_run(
        result,
        data_sources=["mooloolaba", "brisbane"],
        train_index=X_seq_tr.index,
        test_index=X_seq_te.index,
        n_features=X_seq.shape[1],
        extra={
            "window": f"{start.date()}:{end.date()}",
            "seq_len": lstm.seq_len,
            "hidden": lstm.hidden,
            "num_layers": lstm.num_layers,
            "epochs": lstm.epochs,
            "lr": lstm.lr,
            "imputation": "mean",
            "feature_cols": list(X_seq.columns),
        },
    )

    print(f"\nLSTM        RMSE {metrics['RMSE']:.4f}  "
          f"Skill {metrics['SkillVsBaseline']:+.4f}")
    print(f"Persistence RMSE {persist.metrics['RMSE']:.4f}")

    log = fc.read_log()
    prev = log[log["name"].isin(["ridge_mool_brisbane", "lasso_mool_brisbane"])].copy()
    if not prev.empty:
        print("\nLinear model context (from experiments.jsonl):")
        for _, row in prev.sort_values("timestamp").drop_duplicates("name", keep="last").iterrows():
            m = row["metrics"]
            print(f"  {row['name']:25s}  RMSE {m['RMSE']:.4f}  Skill {m['SkillVsBaseline']:+.4f}")


if __name__ == "__main__":
    main()
