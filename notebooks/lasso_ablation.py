"""Per-source Lasso ablation for the README table.

Run:  ./.venv/bin/python notebooks/lasso_ablation.py

For each configuration below, fits a Lasso(alpha=0.001) on the same train/test
split as the linear playground (chronological 80/20 on the 2015-2024 window,
mean imputation, robust scaling) and records RMSE, skill vs. persistence,
non-zero coefficient count, and the single largest-|coef| feature.

The output is a Markdown table written verbatim to stdout, ready to paste into
the README's "Lasso: incremental value of each data source" section.

Logging to ``experiments.jsonl`` is OFF by default — the table is fully
reproducible from this script's stdout, and re-running for verification (e.g.
after a constants change or AEST audit) would otherwise stack 9 identical
rows per run into the log. Flip ``LOG_TO_JSONL = True`` if you want a fresh
sweep to land in the log alongside the playground runs.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

import forecast as fc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

PRIMARY = "mooloolaba"
YEAR_MIN = None
YEAR_MAX = 2024
TEST_FRAC = 0.2
LASSO_KW = {"alpha": 0.001, "max_iter": 10000}
LOG_TO_JSONL = False

# label → (neighbours, wind_stations). Empty lists = source disabled.
CONFIGS: list[tuple[str, list[str], list[str]]] = [
    ("Mooloolaba only",                    [],                             []),
    ("+ Caloundra",                        ["caloundra"],                  []),
    ("+ Brisbane",                         ["brisbane"],                   []),
    ("+ Gold Coast",                       ["gold-coast"],                 []),
    ("+ North Moreton Bay",                ["north-moreton-bay"],          []),
    ("+ Tweed Heads",                      ["tweed-heads"],                []),
    ("+ Palm Beach",                       ["palm-beach"],                 []),
    ("+ Gold Coast + Palm Beach",          ["gold-coast", "palm-beach"],   []),
    ("+ wind (4 stations)",                [],                             ["mountain-creek", "deception-bay",
                                                                            "southport", "lytton"]),
]


def _load_wave() -> pd.DataFrame:
    return fc.restrict_to_years(fc.load_data(buoy=PRIMARY), YEAR_MIN, YEAR_MAX)


def _build_features(
    merged: pd.DataFrame,
    neighbour_cols: list[str],
    wind: pd.DataFrame | None,
) -> pd.DataFrame:
    primary_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
    X = fc.build_buoy_features(primary_only)
    if neighbour_cols:
        X = fc.add_neighbour_features(X, merged, neighbour_cols)
    if wind is not None:
        wind_cols = [c for c in wind.columns if not c.endswith("_deg")]
        X = fc.add_neighbour_features(X, wind, wind_cols)
    return X


def run_one(
    label: str, neighbours: list[str], wind_stations: list[str],
) -> dict:
    wave = _load_wave()
    nb = fc.load_neighbours(wave.index, neighbours)
    wind = fc.load_wind(wave.index, wind_stations)
    wave, nb, wind = fc.restrict_to_overlap(wave, nb, wind)

    merged, neighbour_cols, _ = fc.assemble_inputs(wave, nb, wind)
    X = _build_features(merged, neighbour_cols, wind)
    y = fc.make_target(wave)

    X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y, test_frac=TEST_FRAC)
    preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_tr)
    X_tr_sc = preproc.transform(X_tr)
    X_te_sc = preproc.transform(X_te)
    X_tr = X_tr[preproc.kept_columns_]  # for column-name reporting below

    persist_X = wave[["hsig_m"]]
    pX_tr, pX_te, _, _ = fc.chronological_split(persist_X, y, test_frac=TEST_FRAC)
    persist = fc.evaluate(
        fc.PersistenceForecaster(), pX_tr, y_tr, pX_te, y_te, name="persistence",
    )
    baseline_preds = persist.predictions

    sources = [PRIMARY] + neighbours + wind_stations
    result = fc.evaluate_and_log(
        Lasso(**LASSO_KW),
        X_tr_sc, y_tr, X_te_sc, y_te,
        name=f"ablation_lasso_{label.replace(' ', '_').replace('+', 'plus')}",
        baseline_preds=baseline_preds,
        data_sources=sources,
        extra={
            "ablation_label": label,
            "n_features": X.shape[1],
            "n_neighbours": len(neighbours),
            "n_wind_stations": len(wind_stations),
        },
        log=LOG_TO_JSONL,
    )

    coef = result.model.coef_
    n_nonzero = int((coef != 0).sum())
    top_idx = int(np.argmax(np.abs(coef)))
    top_feat = X_tr.columns[top_idx]

    return {
        "label": label,
        "rmse_cm": result.metrics["RMSE"] * 100,
        "skill": result.metrics["SkillVsBaseline"] * 100,
        "n_nonzero": n_nonzero,
        "n_total": len(coef),
        "top_feat": top_feat,
    }


def main() -> None:
    rows: list[dict] = []
    for label, neighbours, wind in CONFIGS:
        print(f"--- {label} ---")
        row = run_one(label, neighbours, wind)
        print(
            f"  RMSE {row['rmse_cm']:.1f} cm   Skill {row['skill']:+.1f}%   "
            f"nonzero {row['n_nonzero']}/{row['n_total']}   top={row['top_feat']}"
        )
        rows.append(row)

    print("\n\n" + "=" * 60)
    print("README-ready Markdown:\n")
    print("| Data sources | RMSE (cm) | Skill | Non-zero coefs | Top feature |")
    print("|--------------|-----------|-------|----------------|-------------|")
    for r in rows:
        print(
            f"| {r['label']} | {r['rmse_cm']:.1f} | {r['skill']:+.1f}% | "
            f"{r['n_nonzero']} / {r['n_total']} | `{r['top_feat']}` |"
        )


if __name__ == "__main__":
    main()
