# Surf Height Prediction 2

Predicts significant wave height (`hsig_m`) 12 hours ahead at the Mooloolaba wave buoy, Queensland, using data from the Queensland Government open-data buoy network (2015–2025). Neighbour buoys (Brisbane, Caloundra, Gold Coast, North Moreton Bay) are used as additional input features where their histories overlap.

Four installed Python packages back it:

- **`wave_data`** — ETL. Downloads per-buoy yearly records from the CKAN Datastore API, unifies schema, and writes a cleaned CSV on a 30-minute grid.
- **`wind_data`** - ETL.
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

A parallel `wind_data` package fetches hourly 10 m wind from QLD AWS stations on the same CKAN portal:

```bash
# Default: Mountain Creek 2015-2024 → data/mountain-creek_wind_data_2015-2024.csv
./.venv/bin/python -m wind_data

# Any supported station
./.venv/bin/python -m wind_data --station mountain-creek
./.venv/bin/python -m wind_data --station deception-bay
```

Supported stations: `mountain-creek` (Sunshine Coast, effectively co-located with the Mooloolaba buoy) and `deception-bay` (Moreton Bay, ~50 km south of Mooloolaba).

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
| `mooloolaba_brisbane_lstm.py` | LSTM on the same window. Convergence findings across several architecture configs documented in the script header (~25 min CPU per run). |
| `buoy_eda.py` | Multi-buoy EDA: coverage, distributions, seasonality, direction, cross-source correlation. |

## Results

All runs use a chronological 80/20 split. Skill score is vs. persistence on the same test window. Persistence baselines differ across windows (the full-history split, the 2024-2025 neighbour-buoy overlap, and the 2015-2024 wind-overlap window) so RMSEs are only directly comparable within the same window.

| Model | Data sources | Window | RMSE | Skill |
|-------|-------------|--------|------|-------|
| Persistence (baseline) | Mooloolaba | 2015-2025 | 0.289 | — |
| Ridge | Mooloolaba | 2015-2025 | 0.265 | +11.3% |
| HGB (persistence-residual target) | Mooloolaba | 2015-2025 | 0.282 | +5.2% |
| Ridge + HGB ensemble | Mooloolaba | 2015-2025 | 0.277 | +8.2% |
| Lasso | Mooloolaba + Brisbane | 2015-2025 | 0.267 | +15.5% |
| LSTM (15 epochs, seq_len=48, 2 layers) | Mooloolaba + Brisbane | 2015-2025 | 0.362 | −55.2% |
| Persistence (baseline) | Mooloolaba | 2015-2024 | 0.265 | — |
| Ridge | Mooloolaba | 2015-2024 | 0.253 | +9.0% |
| Ridge + Mountain Creek wind | Mooloolaba + Mountain Creek AWS | 2015-2024 | 0.248 | +12.9% |
| Lasso + Mountain Creek wind | Mooloolaba + Mountain Creek AWS | 2015-2024 | 0.248 | +12.6% |
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

## Data sources

Queensland Government open data portal. Fetched via the CKAN Datastore API (`datastore_search`) rather than raw CSV downloads, so resource IDs remain stable across portal file renames. https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation

- **Wave buoy network.** Mooloolaba (2015–2025), Brisbane (2015–2025), Caloundra (2024–2025), Gold Coast (2024–2025), North Moreton Bay (2010–2025).

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


- **Air-quality / meteorology AWS network.** Mountain Creek (2015–2024) — Sunshine Coast station at -26.69, 153.10, with a 10 m ultrasonic wind sensor. Deception Bay (2015–2024) — Moreton Bay station ~50 km south. Both carry the same 10 m wind schema. Pollutant fields are dropped at clean time; only `wind_dir_deg`, `wind_speed_ms`, and the two dispersion stats are kept.

Mountain Creek (Sunshine Coast Council AWS at -26.69, 153.10 — effectively
co-located with the Mooloolaba wave buoy) carries hourly 10 m wind speed and
direction back to 2015. The wave history is sliced to 2015-2024 to match the
wind window — a separate persistence baseline is computed on that same window
so skill scores are directly comparable to the wind-augmented runs (the
existing 2015-2025 persistence row in experiments.jsonl is on a different
test split).

The wind frame is reindexed onto the 30-min wave grid by forward-fill: each
30-min slot inherits the most recent past hourly reading (e.g. the 14:30
slot gets the 14:00 wind value), which is strictly past-only.

Wind direction is circular (359° and 1° are 2° apart, not 358°), so it is
sin/cos-encoded before being passed to add_neighbour_features — same pattern
as the wave-buoy peak_dir_deg encoding in build_mooloolaba_features.


## License

See [LICENSE](LICENSE).



## todo
  2. Check if bias is conditional — plot residuals vs. predicted value or vs. swell period/direction. If bias is concentrated at high wave heights, your model may be      
  underfit there. Adding features like Hs² or interaction terms could help.                                                                                                
  3. Check the target distribution — if large waves are rare in training data, the model learned to hedge toward the mean. Log-transforming Hs before fitting (then
  exponentiating predictions) can reduce this regression-to-the-mean effect. 