# Surf-Height-Prediction-2

Predicts significant wave height (`hsig_m`) 12 hours ahead at the Mooloolaba wave buoy, Queensland, from the Queensland Government open-data buoy feed (2015–2025).

Three installed Python packages back it:

- **`wave_data`** — ETL. Downloads every year from the CKAN Datastore API, unifies schema, and writes a cleaned CSV on a 30-minute grid.
- **`forecast`** — modelling. Target construction, chronological splits, feature engineering, baselines, metrics, an evaluation harness, and sequence-model forecasters (RNN / GRU / LSTM / TCN) built on PyTorch.
- **`viz`** — source-agnostic plotting. Time series (incl. multi-source overlays), correlation heatmaps (feature × horizon, lookback × horizon, cross-source), and model-comparison / residual-analysis diagnostics.

Notebooks in `notebooks/` drive experimentation on top of these packages.

## Problem

Given buoy observations up to time *t* (30-minute cadence), predict `hsig_m` at *t + 12h* — 24 steps ahead. Evaluation is a chronological 80/20 split; the headline metric is **skill score versus persistence** (a positive score means the model actually added information over "it'll be the same as now").

At 12h the autocorrelation of `hsig_m` is ≈ 0.81, so persistence is a stiff baseline.

## Setup

Requires Python 3.14. Create a venv, install pinned deps, install the packages in editable mode:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The `data/` directory is gitignored — populate it by running the pipeline.

## Running the pipeline

Download every year, clean, and export a unified CSV to `data/mooloolaba_wave_data_2015-2025.csv` (~190k rows, a few minutes over the CKAN Datastore API):

```bash
python -m wave_data
```

Or to a custom location:

```bash
python -m wave_data --output path/to/out.csv
```

## Modelling

All modelling code lives in `src/forecast/` and exposes a flat import surface:

```python
import forecast as fc

df = fc.load_data()                              # tz-aware, 30-min grid
y = fc.make_target(df)                           # y.loc[t] == hsig_m[t + 12h]
X = fc.encode_circular(df)                       # sin/cos wave direction
X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y)

result = fc.evaluate(
    fc.LSTMForecaster(seq_len=48, hidden=32, epochs=5),
    X_tr, y_tr, X_te, y_te, name="lstm",
)
print(result.metrics)  # {"MAE": ..., "RMSE": ..., "Bias": ...}
```

Available forecasters:

| family          | classes                                                            |
|-----------------|--------------------------------------------------------------------|
| baselines       | `PersistenceForecaster`, `SeasonalNaiveForecaster`, `ClimatologyHourForecaster` |
| linear / tree   | any scikit-learn regressor (OLS, Ridge, RF, GB, …)                 |
| sequence models | `SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster` |

`fc.evaluate(...)` fits on NaN-masked training rows, predicts on `X_test` (NaN-safe for sklearn models too), and returns metrics + predictions aligned to the test index. `fc.compare([r1, r2, ...])` stacks several results into a sorted DataFrame.

## Logging experiments

Runs can be appended to a JSONL log at `experiments.jsonl` (repo root, committed) so results are durable across sessions and reproducible by git SHA. `fc.evaluate_and_log(...)` is a drop-in replacement for `fc.evaluate(...)`:

```python
result = fc.evaluate_and_log(
    Ridge(alpha=1.0),
    X_tr, y_tr, X_te, y_te,
    name="ridge",
    baseline_preds=persistence_preds,
    data_sources=["mooloolaba"],
)
```

For results computed outside the harness (custom fitting loops, ensembles, residual targets), build the `EvaluationResult` yourself and call `fc.log_run(result, ...)`. Each record captures `{timestamp, git_sha, name, model_class, hyperparams, data_sources, n_features, train, test, metrics, extra}`; the `git_sha` carries a `-dirty` suffix when the working tree has uncommitted changes, so any row can be reproduced by checkout.

Read it back as a DataFrame with `fc.read_log()`.

## Notebooks

```bash
jupyter notebook notebooks/
```

- **`visualization.ipynb`** — data-exploration notebook. Runs in ~10 s end-to-end (no sequence-model training), so `Run All` is cheap. Time-series overviews, channel distributions, missing-value inspection, autocorrelation, feature × horizon and lookback × horizon correlation heatmaps, and cheap-model residual diagnostics. All plots go through `viz`.
- **`forecast_comparison.ipynb`** — the main modelling surface. Loads the unified CSV, engineers lag/rolling features, fits baselines, linear, tree, and sequence models, and presents a full skill-score comparison plus error analysis.
- `wave_data_unification.ipynb` — legacy; superseded by `python -m wave_data`.

## Running tests

```bash
pytest src/tests/ -v
```

Network calls are mocked, so tests run offline.

## Project structure

```
Surf-Height-Prediction-2/
├── src/
│   ├── wave_data/              # ETL package — installed as `wave_data`
│   │   ├── __main__.py         # `python -m wave_data` CLI entry point
│   │   ├── constants.py        # CKAN resource IDs, rename maps, sentinel value
│   │   ├── downloader.py       # per-year CKAN Datastore fetch + schema normalisation
│   │   └── pipeline.py         # unify / clean / run — exports the cleaned CSV
│   ├── forecast/               # modelling package — installed as `forecast`
│   │   ├── config.py           # HORIZON_STEPS, TARGET_COL, FEATURE_COLS, CIRCULAR_COLS
│   │   ├── data.py             # load_data, make_target, chronological_split
│   │   ├── features.py         # lag, rolling, cyclical time, circular direction encoding
│   │   ├── baselines.py        # Persistence, SeasonalNaive, ClimatologyHour forecasters
│   │   ├── neural.py           # SimpleRNN / GRU / LSTM / TCN forecasters (PyTorch)
│   │   ├── metrics.py          # MAE, RMSE, bias, skill score (all NaN-aware)
│   │   ├── evaluate.py         # fit / predict / score harness + `compare` helper
│   │   └── experiments.py      # append-only JSONL run log + `read_log` reader
│   ├── viz/                    # plotting package — installed as `viz`
│   │   ├── timeseries.py       # plot_series, plot_multi_source, autocorrelation_curve
│   │   ├── correlation.py      # feature × horizon, lookback × horizon, cross-source heatmaps
│   │   └── diagnostics.py      # rmse_bar, residual_timeseries, residual_by_bin
│   └── tests/                  # pytest suite (network-mocked)
│       ├── test_downloader.py
│       ├── test_pipeline.py
│       └── test_forecast.py
├── notebooks/
│   ├── visualization.ipynb         # cheap data-exploration notebook (no training)
│   ├── forecast_comparison.ipynb   # main modelling notebook
│   └── wave_data_unification.ipynb # legacy
├── data/                       # gitignored — generated by `python -m wave_data`
├── experiments.jsonl           # append-only run log (committed; one JSON record per run)
├── CLAUDE.md                   # non-obvious behaviour, invariants, gotchas
├── pyproject.toml              # package metadata; editable install target
└── requirements.txt            # full pinned env (regen with `pip freeze`)
```

**Package layout rationale.** `wave_data` (ETL), `forecast` (modelling), and `viz` (plotting) are deliberately separated so that downstream code can import a trained forecaster without pulling in HTTP / CKAN dependencies, plotting can be applied to any data source (buoy, atmospheric reanalysis, another buoy) without coupling to the modelling code, and the data pipeline can be swapped (e.g. for a different buoy) without touching the models. All three live under `src/` with an editable install so notebooks and scripts share the same import path without `sys.path` hacks.

**Multi-source expectations.** `viz` accepts a `dict[str, pd.Series | pd.DataFrame]` keyed by source label (buoy name, reanalysis product, …) wherever multi-source comparisons make sense — time-series overlays, cross-source correlation matrices. Future additions (e.g. a sister buoy, BOM wind fields) plug in by contributing a loader that produces a DataFrame with the shared time axis and calling the same `viz` functions.

**Non-obvious architecture points** (invariants and gotchas) are documented in [`CLAUDE.md`](CLAUDE.md).

## Dataset schema

The unified CSV has a `datetime_utc` index at 30-minute intervals (raw records are AEST; `pipeline.clean` localises then converts to UTC for storage):

| Column | Description |
|--------|-------------|
| `hsig_m` | Significant wave height (metres) |
| `hmax_m` | Maximum wave height (metres) |
| `tz_s` | Zero-crossing period (seconds) |
| `tp_s` | Peak period (seconds) |
| `peak_dir_deg` | Peak wave direction (degrees) |
| `sst_c` | Sea surface temperature (°C) |

Missing or erroneous readings are encoded as `-99.9` in the raw source files; `pipeline.clean` replaces them with `NaN` and reindexes onto a gap-free 30-minute grid.

## Data source

Queensland Government open data portal, Mooloolaba wave buoy. Fetched via the CKAN Datastore API (`datastore_search`) rather than by downloading raw CSVs, so resource IDs remain stable across portal file renames.

## License

See [LICENSE](LICENSE).
