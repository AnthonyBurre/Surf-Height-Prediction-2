# Surf-Height-Prediction-2

Predicts significant wave height (`hsig_m`) 12 hours ahead at the Mooloolaba wave buoy, Queensland, using data from the Queensland Government open-data buoy network (2015–2025). Neighbour buoys (Brisbane, Caloundra, Gold Coast, North Moreton Bay) are used as additional input features where their histories overlap.

Three installed Python packages back it:

- **`wave_data`** — ETL. Downloads per-buoy yearly records from the CKAN Datastore API, unifies schema, and writes a cleaned CSV on a 30-minute grid.
- **`viz`** — source-agnostic plotting. Time series (incl. multi-source overlays), correlation heatmaps (feature × horizon, lookback × horizon, cross-source), and model-comparison / residual-analysis diagnostics.
- **`forecast`** — modelling. Target construction, chronological splits, feature engineering, baselines, metrics, an evaluation harness, and sequence-model forecasters (RNN / GRU / LSTM / TCN) built on PyTorch.


Experiment scripts in `notebooks/` run on top of these packages.

## Problem

Given buoy observations up to time *t* (30-minute cadence), predict `hsig_m` at *t + 12h* — 24 steps ahead. Evaluation is a chronological 80/20 split; the headline metric is **skill score versus persistence** (a positive score means the model added information over "it'll be the same as now").

At 12h the autocorrelation of `hsig_m` is ≈ 0.81, so persistence is a stiff baseline.

## Setup

Requires Python 3.14. Create a venv and install all pinned dependencies (including the editable local packages):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `data/` directory is gitignored — populate it by running the pipeline.

## Running the pipeline

Download, clean, and export a unified CSV (a few minutes over the CKAN Datastore API):

```bash
# Default: Mooloolaba 2015-2025 → data/mooloolaba_wave_data_2015-2025.csv
./.venv/bin/python -m wave_data

# Any supported buoy
./.venv/bin/python -m wave_data --buoy brisbane
./.venv/bin/python -m wave_data --buoy caloundra
```

Supported buoys: `mooloolaba`, `brisbane`, `caloundra`, `gold-coast`, `north-moreton-bay`.

## Modelling

All modelling code lives in `src/forecast/` and exposes a flat import surface:

```python
import forecast as fc

df = fc.load_data()                              # tz-aware, 30-min grid
y  = fc.make_target(df)                          # y.loc[t] == hsig_m[t + 12h]
X  = fc.build_mooloolaba_features(df)            # lag + rolling + momentum + time features
X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y)

result = fc.evaluate_and_log(
    fc.Ridge(alpha=1.0),
    X_tr, y_tr, X_te, y_te,
    name="ridge", data_sources=["mooloolaba"],
)
print(result.metrics)  # {"MAE": ..., "RMSE": ..., "Bias": ..., "SkillVsBaseline": ...}
```

### Feature engineering

`fc.build_mooloolaba_features(df)` produces the full primary-buoy feature matrix (circular encoding, time features, lags, rolling stats, momentum). Neighbour buoys are appended with `fc.add_neighbour_features(X, source_df, columns)`. Both accept a `FeatureConfig` to tune lag steps, rolling windows, and delta steps:

```python
cfg = fc.FeatureConfig(lag_steps=[1, 2, 6, 24], roll_windows=[12, 48])
X = fc.build_mooloolaba_features(df, config=cfg)
```

For sequence models (LSTM / GRU / TCN), use `fc.build_seq_features(df)` — circular encoding and time features only, no pre-built lags (the model windows its own input).

### Available forecasters

| family          | classes                                                                         |
|-----------------|---------------------------------------------------------------------------------|
| baselines       | `PersistenceForecaster`, `SeasonalNaiveForecaster`, `ClimatologyHourForecaster` |
| linear / tree   | any scikit-learn regressor (Ridge, Lasso, HGB, …)                               |
| sequence models | `SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`       |

## Logging experiments

`fc.evaluate_and_log(...)` is a drop-in for `fc.evaluate(...)` that appends a record to `experiments.jsonl` (repo root, committed). For results computed outside the harness (ensembles, custom loops), use `fc.log_run(result, ...)`. Each record captures `{timestamp, git_sha, name, model_class, hyperparams, data_sources, n_features, train, test, metrics, extra}`; the `git_sha` carries a `-dirty` suffix when the working tree has uncommitted changes. Read the log back as a DataFrame with `fc.read_log()`.

## Experiment scripts

All scripts are plain `.py` files — run directly:

```bash
./.venv/bin/python notebooks/<script>.py
```

| script | description |
|--------|-------------|
| `forecast_v2.py` | Full 2015-2025 history. Compares persistence, Ridge, HGB (direct + residual target), and a Ridge/HGB ensemble. Best: Ridge RMSE 0.265, skill +11.3%. |
| `mooloolaba_brisbane_forecast.py` | Ridge + Lasso on the 2015-2025 Mooloolaba + Brisbane overlap window. |
| `mooloolaba_brisbane_lstm.py` | LSTM on the same window. Convergence findings across several architecture configs documented in the script header (~25 min CPU per run). |
| `multi_buoy_forecast.py` | 2024-2025 window, all four neighbour buoys. Ridge + HGB. Key finding: neighbour buoys add ~+6 pp skill (RMSE 0.244, +19.5% vs persistence). |
| `buoy_eda.py` | Multi-buoy EDA: coverage, distributions, seasonality, direction, cross-source correlation. |

## Results

All runs use a chronological 80/20 split. Skill score is vs. persistence on the same test window. The 2024-2025 window is a separate split scoped to the neighbour-buoy overlap period; its persistence baseline (RMSE 0.272) differs from the full-history one (RMSE 0.289).

| Model | Data sources | Window | RMSE | Skill |
|-------|-------------|--------|------|-------|
| Persistence (baseline) | Mooloolaba | 2015-2025 | 0.289 | — |
| Ridge | Mooloolaba | 2015-2025 | 0.265 | +11.3% |
| HGB (persistence-residual target) | Mooloolaba | 2015-2025 | 0.282 | +5.2% |
| Ridge + HGB ensemble | Mooloolaba | 2015-2025 | 0.277 | +8.2% |
| Lasso | Mooloolaba + Brisbane | 2015-2025 | 0.267 | +15.5% |
| LSTM (50 epochs, seq_len=48) | Mooloolaba + Brisbane | 2015-2025 | 0.301 | +3.7% |
| Persistence (baseline) | Mooloolaba | 2024-2025 | 0.272 | — |
| Ridge | Mooloolaba + 3 neighbours | 2024-2025 | 0.244 | +19.5% |
| HGB | Mooloolaba + 3 neighbours | 2024-2025 | 0.258 | +10.0% |

The full set of logged runs is in `experiments.jsonl`.

## Running tests

```bash
./.venv/bin/pytest src/tests/ -v
```

Network calls are mocked, so tests run offline.

## Project structure

```
Surf-Height-Prediction-2/
├── src/
│   ├── wave_data/              # ETL package
│   │   ├── __main__.py         # `python -m wave_data [--buoy NAME]` entry point
│   │   ├── constants.py        # CKAN resource IDs per buoy per year
│   │   ├── downloader.py       # per-year CKAN Datastore fetch + schema normalisation
│   │   └── pipeline.py         # unify / clean / export CSV
│   ├── forecast/               # modelling package
│   │   ├── config.py           # HORIZON_STEPS, TARGET_COL, FEATURE_COLS, CIRCULAR_COLS
│   │   ├── data.py             # load_data, make_target, chronological_split
│   │   ├── features.py         # FeatureConfig, build_mooloolaba_features, add_neighbour_features, build_seq_features
│   │   ├── baselines.py        # Persistence, SeasonalNaive, ClimatologyHour forecasters
│   │   ├── neural.py           # SimpleRNN / GRU / LSTM / TCN forecasters (PyTorch)
│   │   ├── metrics.py          # MAE, RMSE, bias, skill score (all NaN-aware)
│   │   ├── evaluate.py         # fit / predict / score harness + compare helper
│   │   └── experiments.py      # append-only JSONL run log + read_log reader
│   ├── viz/                    # plotting package
│   │   ├── timeseries.py       # plot_series, plot_multi_source, autocorrelation_curve
│   │   ├── correlation.py      # feature × horizon, lookback × horizon, cross-source heatmaps
│   │   └── diagnostics.py      # rmse_bar, residual_timeseries, residual_by_bin
│   └── tests/                  # pytest suite (network-mocked)
├── notebooks/                  # experiment scripts
├── data/                       # gitignored — generated by `python -m wave_data`
├── experiments.jsonl           # append-only run log (committed)
├── pyproject.toml
└── requirements.txt            # pinned env (regen: pip freeze > requirements.txt)
```

**Package layout rationale.** `wave_data`, `forecast`, and `viz` are deliberately separated so a trained forecaster can be imported without pulling in HTTP/CKAN dependencies, plotting works against any data source without coupling to the models, and the pipeline can be swapped without touching either. All three live under `src/` with an editable install so scripts share the same import path without `sys.path` hacks.

## Dataset schema

The unified CSV has a `datetime_utc` index at 30-minute intervals (raw records are AEST; `pipeline.clean` localises then converts to UTC):

| Column | Description |
|--------|-------------|
| `hsig_m` | Significant wave height (metres) |
| `hmax_m` | Maximum wave height (metres) |
| `tz_s` | Zero-crossing period (seconds) |
| `tp_s` | Peak period (seconds) |
| `peak_dir_deg` | Peak wave direction (degrees) |
| `sst_c` | Sea surface temperature (°C) |

Missing or erroneous readings (`-99.9` in raw files) are replaced with `NaN` and the index is resampled onto a gap-free 30-minute grid.

## Data source

Queensland Government open data portal, wave buoy network. Fetched via the CKAN Datastore API (`datastore_search`) rather than raw CSV downloads, so resource IDs remain stable across portal file renames. Supported buoys and their available history: Mooloolaba (2015–2025), Brisbane (2015–2025), Caloundra (2024–2025), Gold Coast (2024–2025), North Moreton Bay (2010–2025).

## License

See [LICENSE](LICENSE).
