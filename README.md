# Surf Height Prediction 2

Predicts significant wave height (`hsig_m`) 12 hours ahead at the Mooloolaba wave buoy, Australia, using data from the Queensland Government open-data buoy network (2015–2025). Neighbour buoys and nearby wind stations are used as additional input features.

Three installed Python packages back it:

- **`qld_ckan`** — ETL. Downloads yearly records from the QLD Government CKAN Datastore API, unifies schema, and writes a cleaned CSV on a per-source grid. Two sub-packages: `qld_ckan.wave` (wave-buoy network, 30-minute grid) and `qld_ckan.wind` (air-quality-station 10 metre wind, hourly grid). Shared transport (retrying session, paginated GET, 404-skip year loop, `unify_frames`) lives at the umbrella level.
- **`viz`** — source-agnostic plotting, organised by pipeline stage. Shared time-series primitives (single-source, multi-source overlays, autocorrelation), post-download EDA heatmaps (feature × horizon, cross-source), and post-modeling diagnostics (model comparison, residual analysis).
- **`forecast`** — modelling. Target construction, chronological splits, feature engineering, baselines, metrics, an evaluation harness, and sequence-model forecasters (RNN / GRU / LSTM / TCN) built on PyTorch.

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

Populate the `data/` directory as desired with these python commands:
```bash
# Default: Mooloolaba 2015-2025 → data/mooloolaba_wave_data_2015-2025.csv
./.venv/bin/python -m qld_ckan wave

# Any supported buoy
./.venv/bin/python -m qld_ckan wave --buoy brisbane
./.venv/bin/python -m qld_ckan wave --buoy caloundra
```

Supported buoys: `mooloolaba`, `brisbane`, `caloundra`, `gold-coast`, `north-moreton-bay`.

```bash
# Default: Mountain Creek 2015-2024 → data/mountain-creek_wind_data_2015-2024.csv
./.venv/bin/python -m qld_ckan wind

# Any supported station
./.venv/bin/python -m qld_ckan wind --station deception-bay
```

Supported stations: `mountain-creek` (Sunshine Coast, effectively co-located with the Mooloolaba buoy) and `deception-bay` (Moreton Bay, ~50 km south of Mooloolaba).

## Modelling

All modelling code lives in `src/forecast/` and exposes a flat import surface:

```python
import forecast as fc

df = fc.load_data(buoy="mooloolaba")             # tz-aware, 30-min grid
y  = fc.make_target(df)                          # y.loc[t] == hsig_m[t + 12h]
X  = fc.build_buoy_features(df)                  # lag + rolling + momentum + time features
X_tr, X_te, y_tr, y_te = fc.chronological_split(X, y)

result = fc.evaluate_and_log(
    fc.Ridge(alpha=1.0),
    X_tr, y_tr, X_te, y_te,
    name="ridge", data_sources=["mooloolaba"],
)
print(result.metrics)  # {"MAE": ..., "RMSE": ..., "Bias": ..., "SkillVsBaseline": ...}
```

### Feature engineering

`fc.build_buoy_features(df)` produces the full primary-buoy feature matrix (circular encoding, time features, lags, rolling stats, momentum) for any QLD wave buoy. Neighbour buoys are appended with `fc.add_neighbour_features(X, source_df, columns)`. Both accept a `FeatureConfig` to tune lag steps, rolling windows, and delta steps:

```python
cfg = fc.FeatureConfig(lag_steps=[1, 2, 6, 24], roll_windows=[12, 48])
X = fc.build_buoy_features(df, config=cfg)
```

For sequence models (LSTM / GRU / TCN), use `fc.build_seq_features(df)` — circular encoding and time features only, no pre-built lags (the model windows its own input).

### Feature scaling

Feature magnitudes span orders of scale — `tp_s` in seconds, `hsig_m` in meters, sin/cos columns already in `[-1, 1]`. Penalised linear models care: an unscaled `alpha` shrinks large-magnitude coefficients unevenly. Scaling is fit on train only and applied to both splits, so no test statistics leak.

```python
X_tr_imp, X_te_imp = fc.mean_impute(X_tr, X_te)
X_tr_s, X_te_s = fc.scale_features(X_tr_imp, X_te_imp, method="robust")  # or "standard"
```

`fc.scale_features` defaults to `RobustScaler` (median/IQR) — wave data is heavy-tailed, so storm-spike outliers would inflate a standard-deviation scale. **Circular `*_sin`/`*_cos` columns are passed through untouched**: they are already in `[-1, 1]`, and scaling a sin/cos pair independently would distort the unit-circle geometry. Tree models (HGB) are scale-invariant and are left on the raw matrix.

Sequence models scale internally instead — `_TorchSeqForecaster` standardises its own input channels and target (fit on train), selectable per the `scaler` argument (`"standard"` mean/std, or `"robust"` median/IQR), so they take the unscaled `build_seq_features` frame directly.

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

All runs use a chronological 80/20 split on the **2015-2024** window — the span where both wind stations have data. Skill score is vs. persistence on the same test split. The table below is a curated cross-section; the full set of logged runs (other windows, feature combinations, and the sequence-model sweep) is in `experiments.jsonl`.

| Model | Data sources | RMSE (cm) | Skill |
|-------|-------------|------|-------|
| Persistence (baseline) | Mooloolaba | 26.5 | — |
| LSTM (seq_len=48, hidden=64, 1 layer, 3 epochs) | Mooloolaba + 4 neighbours + wind | 25.9 | +5.3% |
| GRU (seq_len=48, hidden=64, 1 layer, 2 epochs) | Mooloolaba + 4 neighbours + wind | 23.6 | +20.9% |
| HGB (persistence-residual target) | Mooloolaba + 4 neighbours + wind | 23.6 | +20.9% |
| RNN (seq_len=48, hidden=128, 2 layers, 3 epochs) | Mooloolaba + 4 neighbours + wind | 23.2 | +23.4% |
| Lasso (alpha=0.001) | Mooloolaba + 4 neighbours + wind | 23.1 | +24.2% |
| Ridge (alpha=1.0) | Mooloolaba + 4 neighbours + wind | 23.0 | +24.6% |
| TCN (seq_len=48, channels=(64,), 1 block, 2 epochs) | Mooloolaba + 4 neighbours + wind | 23.0 | +24.9% |
| **NanMean ensemble (Ridge + Lasso + HGB)** | Mooloolaba + 4 neighbours + wind | **22.7** | **+26.6%** |

The sequence models are the best per-class configs from a hyperparameter sweep (`notebooks/seq_sweep.py`): train on `raw` circular-encoded channels and keep epochs low (2-3) — they fit the persistence residual and overfit fast. They now use `scaler="robust"` (median/IQR) for their in-forecaster input/target scaling; relative to the previous `"standard"` (mean/std) scaling that helped the simpler recurrent nets (RNN +2.7, GRU +2.6 skill points) but was a wash for LSTM and the TCN — wave data's heavy tail favours a robust scale, but only where the model wasn't already absorbing it. The linear/tree rows are the best runs of `notebooks/linear_playground.py` on the full 7-source feature set (4 neighbour buoys + 2 wind stations, 263 features), with the linear models trained on robust-scaled features (`fc.scale_features`); the TCN edges out a plain Ridge as the best single model, and a nanmean ensemble of Ridge + Lasso + HGB is still the strongest overall. Adding Caloundra is a wash-to-slight-loss for the recurrent models (RNN/GRU/LSTM) but a small gain for the TCN. Once 2025 wind data lands, the whole project moves to a single 2015-2025 window.

### Lasso: incremental value of each data source

To check that the extra sources actually carry signal, here is a plain `Lasso(alpha=0.001)` trained on the Mooloolaba buoy alone, then on Mooloolaba plus each extra source *in isolation* (not cumulative). All rows share the identical 2015-2024 chronological 80/20 split, so RMSE and Skill are directly comparable; "Non-zero coefs" is how many of the feature columns Lasso kept, and "Top feature" is the largest `|coef|`.

| Data sources | RMSE (cm) | Skill | Non-zero coefs | Top feature |
|--------------|-----------|-------|----------------|-------------|
| Mooloolaba only | 25.3 | +9.1% | 53 / 107 | `hsig_m` |
| + Caloundra | 25.2 | +9.9% | 51 / 120 | `hsig_m` |
| + Brisbane | 24.3 | +16.6% | 59 / 120 | `hsig_m` |
| + Gold Coast | 24.1 | +17.3% | 55 / 120 | `gold-coast_hsig_m` |
| + North Moreton Bay | 25.2 | +10.1% | 53 / 120 | `hsig_m` |
| + wind | 24.6 | +14.2% | 86 / 211 | `hsig_m` |

Every added source helps, but not equally: Brisbane and Gold Coast (the southern, swell-upstream buoys) are worth ~7-8 skill points on their own — Gold Coast even displaces the buoy's own `hsig_m` as the top feature — while Caloundra, despite being the closest neighbour, barely moves the needle. Wind adds a moderate lift and roughly doubles the kept-coefficient count.

## Data source

All data comes from the [Queensland Government open data portal](https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation), fetched via the CKAN Datastore API (`datastore_search`) rather than raw CSV downloads — resource IDs stay stable across portal file renames.

Raw records from both sources are naive AEST; `pipeline.clean` localises then converts to UTC, so every unified CSV carries a gap-free `datetime_utc` index — and the wave and wind frames join on a shared UTC axis with no timezone fiddling.

### Wave buoy network

30-minute cadence. Mooloolaba (2015–2025), Brisbane (2015–2025), Caloundra (2013–2025), Gold Coast (2015–2025), North Moreton Bay (2010–2025). Mooloolaba is the prediction target; the others are neighbour-buoy features where their histories overlap.

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

Hourly cadence, 10 m ultrasonic wind sensors on the QLD air-quality monitoring stations. Mountain Creek (2015–2024) — Sunshine Coast Council station at -26.69, 153.10, effectively co-located with the Mooloolaba wave buoy. Deception Bay (2015–2024) — Moreton Bay station ~50 km south. Both carry the same 10 m wind schema; pollutant and temperature fields are dropped at clean time, leaving:

| Column | Description |
|--------|-------------|
| `wind_dir_deg` | Wind direction (degrees true north) |
| `wind_speed_ms` | Wind speed (meters/second) |
| `wind_sigma_theta_deg` | Wind direction standard deviation (degrees) |
| `wind_speed_std_ms` | Wind speed standard deviation (meters/second) |

The wind frame is reindexed onto the 30-minute wave grid by forward-fill: each 30-minute slot inherits the most recent past hourly reading (the 14:30 slot gets the 14:00 value), which is strictly past-only. Wind direction is circular (359° and 1° are 2° apart, not 358°), so it is sin/cos-encoded before being passed to `add_neighbour_features` — the same pattern as the wave-buoy `peak_dir_deg` encoding in `build_buoy_features`.


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

**Package layout rationale.** `qld_ckan`, `forecast`, and `viz` are deliberately separated so a trained forecaster can be imported without pulling in HTTP/CKAN dependencies, plotting works against any data source without coupling to the models, and the pipeline can be swapped without touching either. All three live under `src/` with an editable install so scripts share the same import path without `sys.path` hacks.


## todo
  2. Check if bias is conditional — plot residuals vs. predicted value or vs. swell period/direction. If bias is concentrated at high wave heights, your model may be      underfit there. Adding features like Hs² or interaction terms could help.                                                                    

### Big-leverage modelling changes

  2. Add uncertainty. Surfline customers care about ranges, not points. Quantile HGB (HistGradientBoostingRegressor(loss="quantile", quantile=q)) for P10/P50/P90 is a 10-line addition; conformalised intervals over Ridge are similar. Right now metrics.summarise returns MAE/RMSE/Bias/Skill — extend with pinball loss / coverage and you  have a publishable model.                         
  4. Physics features the linear models can't discover on their own.
    - Wave power proxy H² · T (one column).                                                                                                                                
    - Swell × wind alignment: wind_speed · cos(wind_dir − peak_dir) — onshore vs offshore makes/breaks surf, and it's a multiplicative interaction Ridge can't recover.    
    - tp / wind_speed as a wind-sea vs groundswell separator.                                                                                                              
  These three columns alone often add ~3–5% skill at our scale.                                                                                                            
  5. Multi-horizon forecasts are where you actually beat persistence. At 12h the autocorr is 0.81 — persistence is brutal. At 24/48/72h it collapses. A single-output      
  direct forecaster at h=12 sells the model short. HORIZON_STEPS in forecast/config.py:11 is centralised (good), but the pipeline only fits one h. Either loop over        
  horizons or use a MultiOutputRegressor — the API on Ridge/HGB is one line. This is the change that would make the project read as "a surf forecast model" rather than "a 
  +12h hsig regressor".                                                                                                                                                    
  6. Honest test harness. A single 80/20 split means the "test set" is implicitly used during model selection across the 42 logged runs. Hold out the last 6 months as a
  true blind set, OR move to an expanding-window CV (sklearn TimeSeriesSplit) for hyperparameter selection. Right now skill comparisons across experiments.jsonl rows      
  aren't statistically clean.

7. hsig_m ≠ surf height. A 2 m hsig from 90° at 14s breaks very differently from 2 m from 150° at 8s on the same beach. The README is honest about this ("Predicts       
  significant wave height"), but I'd at least:              
    - Forecast tp_s and peak_dir_deg jointly (multi-output) so downstream code can run a break-specific transform.                                                         
    - Add partitioned swell — sea vs primary vs secondary — if any of the QLD or BOM resources expose it. Otherwise integrating NOAA WAVEWATCH III hindcast (free, global, 
  ~25km grid) is the highest-ROI external data add. Bimodal swells will never fit a single hsig number.  

8. Recommendation: don't build the target_transform param. Nothing was changed in the repo this turn — the measure-first check did its job and said no. The tail underfit is real, but it's a       
  fat-tailed-residual-meets-MSE problem, not a skew problem, so a transform structurally can't fix it.

What would target it: a loss that stops hedging the tail — quantile/pinball loss (your README's "add uncertainty" todo), or up-weighting big-wave samples in training.