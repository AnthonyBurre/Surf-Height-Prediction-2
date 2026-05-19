# Surf Height Prediction 2

Predicts significant wave height (`hsig_m`) 12 hours ahead at the Mooloolaba wave buoy, Australia, using data from the Queensland Government open-data buoy network (2015–2025). Neighbour buoys and nearby wind stations are used as additional input features.

Three packages under `src/` carry the work:

- **`qld_ckan`** — ETL. Downloads yearly records from the QLD CKAN Datastore API, unifies the schema, and writes a cleaned CSV per source. Sub-packages: `qld_ckan.wave` (wave buoys) and `qld_ckan.wind` (air-quality-station 10 m wind). Shared transport (retrying session, paginated GET, 404-skip year loop, `unify_frames`) lives at the umbrella level.
- **`viz`** — source-agnostic plotting, organised by pipeline stage: shared time-series primitives, post-download EDA heatmaps, and post-modelling diagnostics.
- **`forecast`** — modelling. Target construction, chronological splits, feature engineering, baselines, metrics, an evaluation harness, and PyTorch sequence forecasters (RNN / GRU / LSTM / TCN).

Experiment scripts in `notebooks/` run on top of these packages.

## Objective

Given buoy observations up to time *t* (30-minute cadence), predict `hsig_m` at *t + 12h* — 24 steps ahead. Evaluation is a chronological 80/20 split; the headline metric is **skill score versus persistence** (a positive score means the model added information over "it'll be the same as now").

At 12h the autocorrelation of `hsig_m` is ≈ 0.81, so persistence is a stiff baseline.

## Setup

Suggested Python 3.14. Create a venv and install all pinned dependencies (including the editable local packages):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `data/` directory is gitignored — populate it by running the pipeline.

### Running tests

```bash
./.venv/bin/pytest src/tests/ -v
```

Network calls are mocked, so tests run offline.

## Running the pipeline

Populate `data/` with these commands:

```bash
# Wave — default Mooloolaba 2015-2025 → data/mooloolaba_wave_data_2015-2025.csv
./.venv/bin/python -m qld_ckan wave [--buoy brisbane|caloundra|gold-coast|north-moreton-bay]

# Wind — default Mountain Creek 2015-2024 → data/mountain-creek_wind_data_2015-2024.csv
./.venv/bin/python -m qld_ckan wind [--station deception-bay]
```

## Modelling

`forecast` exposes a flat import surface (`import forecast as fc`): target construction (`make_target` shifts `hsig_m` 24 steps ahead), chronological 80/20 split, feature builders, and an `evaluate_and_log` harness that scores `MAE / RMSE / Bias / SkillVsBaseline` against persistence and appends to `experiments.jsonl`. A typical call:

```python
result = fc.evaluate_and_log(
    fc.Ridge(alpha=1.0), X_tr, y_tr, X_te, y_te,
    name="ridge", data_sources=["mooloolaba"],
)
print(result.metrics)  # {"MAE": ..., "RMSE": ..., "Bias": ..., "SkillVsBaseline": ...}
```

### Feature engineering

The feature matrix is assembled in three layers, each a single call:

1. **Base primary-buoy matrix** (`fc.build_buoy_features`) — circular encoding, hour/doy time features, lags, rolling stats, momentum. Lag/rolling/delta grids are tunable via `FeatureConfig`. For sequence models, `fc.build_seq_features` swaps in raw channels with no pre-built lags (the model windows its own input).
2. **Neighbour buoys** (`fc.add_neighbour_features`) — raw value, lag copies, and rolling mean/std per neighbour column, reusing the same `FeatureConfig`.
3. **Wind stations** — same `add_neighbour_features` call on each wind station's columns. `fc.load_wind` sin/cos-encodes `wind_dir_deg` and station-prefixes every column so cross-station features stay distinguishable.

### Feature scaling

Linear models train on `RobustScaler` (median/IQR) by default — wave data is heavy-tailed, so storm spikes would inflate a standard-deviation scale. Circular `*_sin`/`*_cos` columns pass through untouched. Tree models (HGB) take the raw matrix. Sequence models scale internally (`scaler="robust"` or `"standard"`, fit on train) and take the unscaled `build_seq_features` frame.

### Available forecasters

| family          | classes                                                                         |
|-----------------|---------------------------------------------------------------------------------|
| baselines       | `PersistenceForecaster`, `SeasonalNaiveForecaster`, `ClimatologyHourForecaster` |
| linear / tree   | any scikit-learn regressor (Ridge, Lasso, HGB, …)                               |
| sequence models | `SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`       |

## Logging experiments

`fc.evaluate_and_log(...)` is a drop-in for `fc.evaluate(...)` that appends a record to `experiments.jsonl` (committed at repo root); `fc.log_run(result, ...)` covers results computed outside the harness. Read the log back as a DataFrame with `fc.read_log()`.

## Experiment scripts

All scripts are plain `.py` files — run directly:

```bash
./.venv/bin/python notebooks/<script>.py
```

| script | description |
|--------|-------------|
| `linear_playground.py` | Linear / tree playground (Ridge / Lasso / HGB). One `CONFIG` dict for data window, sources, `FeatureConfig` knobs, HGB residual mode, and an optional nanmean ensemble. |
| `seq_playground.py` | Sequence-model playground (RNN / GRU / LSTM / TCN). Single `CONFIG` dict for data window, sources, raw-vs-engineered feature mode, model class, and hyperparameters; auto-detects device and logs each run. |
| `seq_sweep.py` | Small low-epoch hyperparameter sweep over the RNN / GRU / LSTM forecasters, reusing `seq_playground`'s data loading. Logs every run under the `seqsweep` prefix. |
| `wave_eda.py` | Wave-only EDA across all five buoys: coverage, distributions, seasonality, direction, autocorrelation, cross-source correlation. Saves seven `wave_*` PNGs to `notebooks/figures/`. |
| `wind_eda.py` | Wind-only EDA across the available stations: coverage, time series, autocorrelation, direction roses, station comparison. Saves five `wind_*` PNGs to `notebooks/figures/`. |
| `wave_wind_eda.py` | Joint wave + wind EDA: alignment overview, feature-horizon screening, joint distributions. Saves three `wave_wind_*` PNGs to `notebooks/figures/`. |

## Results

All runs use a chronological 80/20 split on the **2015-2024** window — the span where both wind stations have data. Skill score is vs. persistence on the same test split. The table below is a curated cross-section; find the full set of logged runs (other windows, feature combinations, model sweeps) in `experiments.jsonl`.

| Model | Data sources | RMSE (cm) | Skill |
|-------|-------------|------|-------|
| Persistence (baseline) | Mooloolaba | 26.5 | — |
| LSTM (seq_len=48, hidden=64, 1 layer, 3 epochs, weight_decay=1e-4) | Mooloolaba + 4 neighbours + wind | 24.1 | +17.5% |
| HGB (persistence-residual target) | Mooloolaba + 4 neighbours + wind | 23.6 | +20.9% |
| GRU (seq_len=48, hidden=64, 1 layer, 3 epochs, weight_decay=1e-4) | Mooloolaba + 4 neighbours + wind | 23.2 | +23.6% |
| RNN (seq_len=48, hidden=128, 2 layers, 3 epochs, weight_decay=1e-5) | Mooloolaba + 4 neighbours + wind | 23.2 | +23.9% |
| Lasso (alpha=0.001) | Mooloolaba + 4 neighbours + wind | 23.1 | +24.2% |
| Ridge (alpha=1.0) | Mooloolaba + 4 neighbours + wind | 23.0 | +24.6% |
| TCN (seq_len=48, channels=(64,), 1 block, 2 epochs) | Mooloolaba + 4 neighbours + wind | 23.0 | +24.9% |
| **NanMean ensemble (Ridge + Lasso + HGB)** | Mooloolaba + 4 neighbours + wind | **22.7** | **+26.6%** |

**Notes:**

- **Best single model:** TCN narrowly edges Ridge. **Best overall:** nanmean ensemble of Ridge + Lasso + HGB.
- The linear/tree rows are the best runs of `notebooks/linear_playground.py` on the full 7-source feature set (4 neighbour buoys + 2 wind stations, 263 features), with linear models on robust-scaled features (`fc.scale_features`).
- The sequence rows are the best per-class configs from `notebooks/seq_sweep.py`. They overfit fast — keep epochs at 2–3.
- **Robust vs standard scaler** (in-forecaster, sequence models): robust helped the simpler recurrent nets — RNN +2.7, GRU +2.6 skill points — but was a wash for LSTM and TCN. Wave data's heavy tail favours a robust scale, but only where the model isn't already absorbing it.
- **Adam `weight_decay`** (1e-4 for the gated cells, 1e-5 for the vanilla RNN) enabled an extra training epoch without overfit: +0.5 / +2.7 / +12.2 skill points on RNN / GRU / LSTM. The TCN was already self-regularising via its conv `dropout=0.1` so the same lever does nothing for it.
- Inter-layer `rnn_dropout` was tested but never beat the L=1 winners.
- Adding Caloundra is a wash-to-slight-loss for the recurrent models, but a small gain for the TCN.

### Lasso: incremental value of each data source

To check that the extra sources actually carry signal, here is a plain `Lasso(alpha=0.001)` trained on the Mooloolaba buoy alone, then on Mooloolaba plus each extra source *in isolation* (not cumulative). "Non-zero coefs" is how many of the feature columns Lasso kept, and "Top feature" is the largest `|coef|`.

| Data sources | RMSE (cm) | Skill | Non-zero coefs | Top feature |
|--------------|-----------|-------|----------------|-------------|
| Mooloolaba only | 25.3 | +9.1% | 53 / 107 | `hsig_m` |
| + Caloundra | 25.2 | +9.9% | 51 / 120 | `hsig_m` |
| + Brisbane | 24.3 | +16.6% | 59 / 120 | `hsig_m` |
| + Gold Coast | 24.1 | +17.3% | 55 / 120 | `gold-coast_hsig_m` |
| + North Moreton Bay | 25.2 | +10.1% | 53 / 120 | `hsig_m` |
| + wind | 24.6 | +14.2% | 86 / 211 | `hsig_m` |

Every added source helps, but not equally: Brisbane and Gold Coast (the southern, swell-upstream buoys) are worth ~7-8 skill points on their own — Gold Coast even displaces the buoy's own `hsig_m` as the top feature — while Caloundra, despite being the closest neighbour, barely moves the needle.

## Model performance

Each new year is scored as a true blind set against three pre-committed models - best linear, best neural, best ensemble. The QLD wind 2025 release is the current blocker on the first row.

**Pre-committed candidates for 2025**, all trained on 2015-2024 with Mooloolaba + 4 neighbours + wind:

- **Linear** — Ridge (alpha=1.0)
- **Neural** — TCN (seq_len=48, channels=(64,), 1 block, 2 epochs)
- **Ensemble** — NanMean of Ridge + Lasso + HGB

| Year | Model | RMSE (cm) | Skill |
|------|-------|-----------|-------|
| 2025 | Ridge | _TBD_ | _TBD_ |
| 2025 | TCN | _TBD_ | _TBD_ |
| 2025 | Ensemble | _TBD_ | _TBD_ |

## Data source

All data comes from the [Queensland Government open data portal](https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation), fetched via the CKAN Datastore API (`datastore_search`) rather than raw CSV downloads for stability.

Raw records from both sources are naive AEST; `pipeline.clean` localises then converts to UTC, so every unified CSV carries a gap-free `datetime_utc` index.

**Coverage:** 2015-onward across all sources. Wave runs to 2025; wind runs to 2024, with 2025 to be folded in once QLD publishes it.

### Wave buoy network

30-minute cadence. Mooloolaba is the prediction target; Brisbane, Caloundra, Gold Coast, and North Moreton Bay feed in as neighbour-buoy features where their histories overlap.

Missing or erroneous readings (`-99.9` in the raw files) are replaced with `NaN`.

| Column | Description |
|--------|-------------|
| `hsig_m` | Significant wave height (meters) |
| `hmax_m` | Maximum wave height (meters) |
| `tz_s` | Zero-crossing period (seconds) |
| `tp_s` | Peak period (seconds) |
| `peak_dir_deg` | Peak wave direction (degrees) |
| `sst_c` | Sea surface temperature (°C) |

### Wind (air-quality monitoring network)

Hourly cadence, 10 m ultrasonic wind sensors on the QLD air-quality monitoring stations. Mountain Creek is right by Mooloolaba and Deception Bay is ~50 km south on Moreton Bay. Pollutant and temperature fields are dropped at clean time, leaving:

| Column | Description |
|--------|-------------|
| `wind_dir_deg` | Wind direction (degrees true north) |
| `wind_speed_ms` | Wind speed (meters/second) |
| `wind_sigma_theta_deg` | Wind direction standard deviation (degrees) |
| `wind_speed_std_ms` | Wind speed standard deviation (meters/second) |

The wind frame is reindexed onto the 30-minute wave grid by forward-fill.


## Project structure

```
Surf-Height-Prediction-2/
├── src/
│   ├── qld_ckan/               # ETL package — QLD CKAN Datastore client
│   │   ├── __init__.py         # session, paginate_records, fetch_all_years, unify_frames
│   │   ├── __main__.py         # `python -m qld_ckan {wave,wind} [...]` entry point
│   │   ├── wave/               # wave-buoy network (30-min grid)
│   │   │   ├── constants.py    # CKAN resource IDs per buoy per year
│   │   │   ├── downloader.py   # per-year fetch + pre-2017 column normalisation
│   │   │   └── pipeline.py     # unify / clean / export CSV
│   │   └── wind/               # AWS station 10 m wind (hourly grid)
│   │       ├── constants.py
│   │       ├── downloader.py
│   │       └── pipeline.py
│   ├── forecast/               # modelling package
│   │   ├── config.py           # HORIZON_STEPS, TARGET_COL, FEATURE_COLS, CIRCULAR_COLS
│   │   ├── data.py             # load_data, make_target, chronological_split, load_neighbours, load_wind, restrict_to_overlap
│   │   ├── features.py         # FeatureConfig, build_buoy_features, add_neighbour_features, build_seq_features
│   │   ├── baselines.py        # Persistence, SeasonalNaive, ClimatologyHour forecasters
│   │   ├── neural.py           # SimpleRNN / GRU / LSTM / TCN forecasters (PyTorch)
│   │   ├── metrics.py          # MAE, RMSE, bias, skill score (all NaN-aware)
│   │   ├── evaluate.py         # fit / predict / score harness + compare helper
│   │   └── experiments.py      # append-only JSONL run log + read_log reader
│   ├── viz/                    # plotting package — split by pipeline stage
│   │   ├── timeseries.py       # SHARED:        plot_series, plot_multi_source, autocorrelation_curve
│   │   ├── eda.py              # POST-DOWNLOAD: feature × horizon, cross-source heatmaps
│   │   └── diagnostics.py      # POST-MODELING: rmse_bar, residual_timeseries, residual_by_bin
│   └── tests/                  # pytest suite (network-mocked)
├── notebooks/                  # experiment scripts
├── data/                       # gitignored — generated by `python -m qld_ckan {wave,wind}`
├── experiments.jsonl           # append-only run log (committed)
├── pyproject.toml
└── requirements.txt            # pinned env (regen: pip freeze > requirements.txt)
```

**Package layout rationale.** The three packages are deliberately split so a trained forecaster can be imported without HTTP/CKAN deps, and `viz` stays decoupled from the models. All three live under `src/` via an editable install so scripts share the import path without `sys.path` hacks.

## Future Expansions

1. **Predict quantiles, not just the mean.** Quantile HGB (`HistGradientBoostingRegressor(loss="quantile", quantile=q)`) for P10/P50/P90 is a ~10-line addition; conformalised intervals over Ridge are similar. Also the structural fix for the confirmed tail underfit — residuals-vs-predicted plots show bias concentrated at high wave heights, a fat-tailed-residual-meets-MSE problem that a target transform can't fix. A quantile/pinball loss stops the model hedging the tail.

2. **Multi-horizon forecasts.** `HORIZON_STEPS` in `forecast/config.py:11` is centralised but the pipeline only fits one h. At 12h persistence is brutal (autocorr 0.81); at 24/48/72h it collapses, and that's where a real model has room to win. This is the change that would make the project read as a surf forecast model rather than a +12h `hsig` regressor.

3. **Multi-output forecasts.** A 2 m `hsig` from 90° at 14 s breaks very differently from 2 m from 150° at 8 s on the same beach. Forecast `tp_s` and `peak_dir_deg` jointly (`MultiOutputRegressor`, same API as item 2) so downstream code can run a break-specific transform.

4a. **Partitioned swell.** Sea vs primary vs secondary swell components, if any QLD or BOM resource exposes them. Bimodal swells will never fit a single `hsig` number.

4b. **NOAA WAVEWATCH III hindcast.** Free, global, ~25 km grid — the highest-ROI external data add if partitioned swell is unavailable.