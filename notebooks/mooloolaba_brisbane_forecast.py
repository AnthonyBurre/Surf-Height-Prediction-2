"""Mooloolaba + Brisbane +12h hsig_m forecast — linear models.

Run:  ./.venv/bin/python notebooks/mooloolaba_brisbane_forecast.py

Uses the full 2015-2025 overlap window and an 80/20 chronological split.
For LSTM results on this same data, see mooloolaba_brisbane_lstm.py.

Models
------
1. persistence              — baseline
2. ridge_mool_brisbane      — Ridge, Mooloolaba + Brisbane lag/rolling features
3. lasso_mool_brisbane      — Lasso (same feature matrix, sparse linear)
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge

import forecast as fc
from forecast.metrics import summarise

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


def timed(label, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    print(f"  [{time.time() - t0:5.1f}s] {label}")
    return out


def main() -> pd.DataFrame:
    mool = fc.load_data()
    bris = fc.load_data(DATA_DIR / "brisbane_wave_data_2015-2025.csv")

    start = max(mool.index.min(), bris.index.min())
    end   = min(mool.index.max(), bris.index.max())
    mool  = mool.loc[start:end]
    bris  = bris.loc[start:end]

    print(f"Window : {start.date()} → {end.date()}")
    print(f"Rows   : {len(mool):,}  (Mooloolaba)  {len(bris):,}  (Brisbane)")

    merged = mool.copy()
    merged["brisbane_hsig_m"] = bris["hsig_m"]
    print(f"Brisbane hsig_m NaN in merged frame: {merged['brisbane_hsig_m'].isna().mean()*100:.1f}%\n")

    y    = fc.make_target(merged)
    X_lin = fc.build_mooloolaba_features(mool.loc[start:end])
    X_lin = fc.add_neighbour_features(X_lin, merged, ["brisbane_hsig_m"])
    X_p   = merged[["hsig_m"]]

    X_lin_tr, X_lin_te, y_tr, y_te = fc.chronological_split(X_lin, y)
    X_p_tr,   X_p_te,   _,    _    = fc.chronological_split(X_p,   y)

    print(f"Total rows : {len(merged):,}  |  train: {len(X_lin_tr):,}  test: {len(X_lin_te):,}")
    print(f"Linear features : {X_lin.shape[1]}\n")

    # sst_c goes 100% NaN in the test half of 2025 — mean-impute so Ridge's
    # all-or-nothing NaN mask doesn't drop the entire test period.
    X_lin_tr_imp, X_lin_te_imp = mean_impute(X_lin_tr, X_lin_te)

    results: list[fc.EvaluationResult] = []
    window_str = f"{start.date()}:{end.date()}"

    print("=== Baseline: persistence ===")
    persist = timed(
        "persistence_mool_brisbane",
        fc.evaluate_and_log, fc.PersistenceForecaster(),
        X_p_tr, y_tr, X_p_te, y_te,
        name="persistence_mool_brisbane",
        data_sources=["mooloolaba"],
        extra={"window": window_str},
    )
    results.append(persist)
    pp = persist.predictions

    print("\n=== Ridge ===")
    ridge = timed(
        "ridge_mool_brisbane",
        fc.evaluate_and_log, Ridge(alpha=1.0),
        X_lin_tr_imp, y_tr, X_lin_te_imp, y_te,
        name="ridge_mool_brisbane",
        baseline_preds=pp,
        data_sources=["mooloolaba", "brisbane"],
        extra={"window": window_str, "imputation": "mean"},
    )
    results.append(ridge)

    print("\n=== Lasso ===")
    lasso = timed(
        "lasso_mool_brisbane",
        fc.evaluate_and_log, Lasso(alpha=0.001, max_iter=5000),
        X_lin_tr_imp, y_tr, X_lin_te_imp, y_te,
        name="lasso_mool_brisbane",
        baseline_preds=pp,
        data_sources=["mooloolaba", "brisbane"],
        extra={"window": window_str, "imputation": "mean", "alpha": 0.001},
    )
    results.append(lasso)

    lasso_coef = lasso.model.coef_
    n_nonzero  = int((lasso_coef != 0).sum())
    print(f"  Non-zero coefficients: {n_nonzero} / {len(lasso_coef)}")
    top_lasso = (
        pd.Series(np.abs(lasso_coef), index=X_lin_tr_imp.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    print("  Top 15 features by |coefficient|:")
    for feat, val in top_lasso.items():
        print(f"    {feat:40s}  {val:.4f}")

    print("\n=== Results ===")
    table = fc.compare(results).round(4)
    print(table.to_string())

    log = fc.read_log()
    prev = log[log["name"].isin(["ridge", "ridge_multi_2024"])].copy()
    if not prev.empty:
        print("\nContext from experiments.jsonl:")
        for _, row in prev.sort_values("timestamp").drop_duplicates("name", keep="last").iterrows():
            m = row["metrics"]
            print(f"  {row['name']:25s}  RMSE {m['RMSE']:.4f}  Skill {m['SkillVsBaseline']:+.4f}")

    return table


if __name__ == "__main__":
    main()
