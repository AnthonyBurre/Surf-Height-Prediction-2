# Surf Height Prediction 2

An exercise in predictive modeling, this project is all about forecasting significant wave height (`hsig_m`) as measured by the Mooloolaba wave buoy off of the sunny coast in Queensland, Australia. All data in this project comes from the Queensland Government open-data api, which provides us with several wind and wave monitoring stations in the region.



Three packages under `src/`:

- **`qld_ckan`** — ETL. Downloads yearly records from the QLD CKAN Datastore API, unifies the schema, and writes a cleaned CSV per source. Sub-packages: `qld_ckan.wave` (wave buoys) and `qld_ckan.wind` (air-quality-station 10 m wind). Shared transport (retrying session, paginated GET, 404-skip year loop, `unify_frames`) lives at the umbrella level.
- **`viz`** — source-agnostic plotting, organised by pipeline stage: shared time-series primitives, post-download EDA heatmaps, and post-modelling diagnostics.
- **`forecast`** — modelling. Target construction, chronological splits, feature engineering, baselines, metrics, an evaluation harness, and PyTorch sequence forecasters (RNN / GRU / LSTM / TCN).

Experiment scripts in `notebooks/` run on top of these packages.

## Objective

Given buoy observations up to time *t* (30-minute cadence), predict `hsig_m` at *t + 12h* — 24 steps ahead. Evaluation is a chronological 80/20 split; the headline metric is **skill score versus persistence** (a positive score means the model added information over "it'll be the same as now").

At 12h the autocorrelation of `hsig_m` is ≈ 0.81, so persistence is a stiff baseline.

## Setup

With [uv](https://docs.astral.sh/uv/) installed:
```bash
uv sync --all-extras
```

The `data/` directory is gitignored — generate it by running the pipeline.

### Running tests

```bash
./.venv/bin/pytest src/tests/ -v
```

Network calls are mocked, so tests run offline.

## Data sources

All data comes from the [Queensland Government open data portal](https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation), fetched via the CKAN Datastore API (`datastore_search`) rather than raw CSV downloads for stability. Resource IDs are stable even when the portal renames the underlying file.

Raw records from both sources are naive AEST; `pipeline.clean` localises to Australia/Brisbane (UTC+10, no DST) and that is the canonical project timezone — every unified CSV carries a gap-free `datetime` index in Brisbane time. `df.index.year` therefore returns the source-data year directly. The `forecast.SOURCE_TZ` constant is exported for any downstream code that needs to convert to UTC for a cross-source join (e.g. BOM/GFS reanalysis grids, which are UTC-native).

The CKAN catalogue also hosts multi-year *historical* bundles for several buoys (Mooloolaba 2000-2014, Brisbane 1976-2011, Gold Coast 1987-2014, Tweed Heads 1995-2011). These sample at a non-30-minute cadence (1 h to 12 h, with drifting minute offsets) and are therefore **excluded from the registry** — the project's grid is a strict 30-minute wave / 1-hour wind axis. The two bundles that *are* on the standard grid (North Moreton Bay 2010-2015, Caloundra 2013-2015) are kept and define the earliest available years for those buoys.

### Wave buoy network

30-minute cadence. Mooloolaba is the prediction target; Brisbane, Caloundra, Gold Coast, North Moreton Bay, Palm Beach, Tweed Heads, and Wide Bay feed in as neighbour-buoy features where their histories overlap. Missing or erroneous readings (`-99.9` in the raw files) are replaced with `NaN`.

| Column | Description |
|--------|-------------|
| `hsig_m` | Significant wave height (meters) |
| `hmax_m` | Maximum wave height (meters) |
| `tz_s` | Zero-crossing period (seconds) |
| `tp_s` | Peak period (seconds) |
| `peak_dir_deg` | Peak wave direction (degrees) |
| `sst_c` | Sea surface temperature (°C) |

### Wind (air-quality monitoring network)

Hourly cadence, 10 m ultrasonic wind sensors on the QLD air-quality monitoring stations. Mountain Creek pairs with the Mooloolaba buoy, Deception Bay sits ~50 km south on Moreton Bay, Lytton is at the mouth of the Brisbane River (paired with the Brisbane buoy), and Southport sits on the Gold Coast (paired with the Gold Coast / Palm Beach buoys). Pollutant and temperature fields are dropped at clean time, leaving:

| Column | Description |
|--------|-------------|
| `wind_dir_deg` | Wind direction (degrees true north) |
| `wind_speed_ms` | Wind speed (meters/second) |
| `wind_sigma_theta_deg` | Wind direction standard deviation (degrees) |
| `wind_speed_std_ms` | Wind speed standard deviation (meters/second) |

The wind frame is reindexed onto the 30-minute wave grid by forward-fill.

### Running the pipeline

Populate `data/` with these commands (one CSV per source):

```bash
# Wave — default Mooloolaba 2015-2025 → data/mooloolaba_wave_data_2015-2025.csv
./.venv/bin/python -m qld_ckan wave [--buoy brisbane|caloundra|gold-coast|north-moreton-bay|palm-beach|tweed-heads|wide-bay]

# Wind — default Mountain Creek 2010-2024 → data/mountain-creek_wind_data_2010-2024.csv
./.venv/bin/python -m qld_ckan wind [--station deception-bay|lytton|southport]
```

Both subcommands accept `--year-min` / `--year-max` (inclusive) to clip the registry before download — handy for a quick experiment on a sub-window without re-fetching the full history. Bounds filter on the resource-id dict's year keys; the output filename reflects the filtered range.

```bash
# Brisbane buoy, 2018-2020 only → data/brisbane_wave_data_2018-2020.csv
./.venv/bin/python -m qld_ckan wave --buoy brisbane --year-min 2018 --year-max 2020
```

### Coverage

The per-source / per-year completeness grids below are produced by `notebooks/wave_eda.py` and `notebooks/wind_eda.py`. Grey cells mean the station wasn't deployed yet; red cells flag partial years (deployment mid-year, sensor outages, the wide-bay 2019-2021 sparsity).

![Wave coverage](notebooks/figures/wave_coverage.png)

![Wind coverage](notebooks/figures/wind_coverage.png)

### Dataset selection: short vs long runtime

The coverage grids define the trade space for any experiment. The choice is a breadth-vs-depth call across two axes — how far back to train, and how many neighbour sources to include:

1. **Long window, narrow set (2010-2024 AEST, Mooloolaba target + 3 neighbours, 2 wind stations).** North Moreton Bay (2010 bundle), plus Mountain Creek + Deception Bay wind cover the full window; Brisbane and Tweed Heads join from 2012, Caloundra from 2013, Gold Coast from 2014. The Mooloolaba target itself starts 2015 — so practically this option is "Mooloolaba 2015-2024 with pre-2015 neighbour context", useful for sequence models whose lookback can stretch back further than the target.
2. **Standard window, full neighbour set (2015-2024 AEST, 5 buoys + 3 wind stations).** The default for the headline results. Lytton wind starts 2015, so this window is the deepest one where every "always-on" source has data. Mooloolaba + Brisbane / Caloundra / Gold Coast / North Moreton Bay / Tweed Heads neighbours and Mountain Creek / Deception Bay / Lytton wind — 10 years of training, no late-deployment gaps.
3. **Short window, wide set (2019-2024 AEST, 7 buoys + 4 wind stations).** Palm Beach (deployed 2017), Southport wind (mid-2018), and Wide Bay (2019, the only buoy upstream of northerly swells) only join here. Six years of training in exchange for two extra neighbour buoys and one extra wind station.

Each option assumes the same chronological 80/20 split and the same +12 h horizon. The Results section below runs (2) as the headline and includes a (2)-vs-(3) comparison; (1) is on the table for any experiment that benefits from extra pre-2015 neighbour history without needing a wider source set. Choice of window is a `restrict_to_years(...)` call in the script, not a re-download — every CSV in `data/` already carries its full available range.

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

### Preprocessing pipeline

`forecast.preprocess.Preprocessor` bundles the three steps the playgrounds run between `chronological_split` and `model.fit`:

1. **Drop sparse columns** — any column whose **train-set** NaN fraction exceeds `max_nan_frac` (default 0.5) is removed. Mean-imputing a 90%-NaN column gives a near-constant feature that silently corrodes gradient-based sequence models: every window contains the imputed value, the gradient is dominated by it, but it carries no real signal (e.g. Lytton's `wind_speed_std_ms`). The `wave_column_coverage.png` and `wind_column_coverage.png` EDA figures surface candidates ahead of time.
2. **Mean impute** — column-wise mean from training data fills remaining NaNs.
3. **Scale** (optional) — `"robust"` (median/IQR, default for linear models) or `"standard"`; `*_sin`/`*_cos` columns pass through untouched. Sequence models scale internally, so they skip this step (`scaling=None`).

The standalone helpers (`fc.drop_sparse_columns`, `fc.mean_impute`, `fc.scale_features`) still exist for one-shot use, but the playgrounds use the class so the fitted state can be inspected, asserted, and pickled:

```python
preproc = fc.Preprocessor(max_nan_frac=0.5, scaling="robust").fit(X_train)
X_train_p = preproc.transform(X_train)
X_test_p  = preproc.transform(X_test)

# Held-out year scoring: pair the model with its preprocessor.
preproc.save("models/ridge_preproc.pkl")
# Later, with a new year's raw inputs:
preproc = fc.Preprocessor.load("models/ridge_preproc.pkl")
X_2025_p = preproc.transform(X_2025)  # raises if any fit-time column is missing
```

`transform()` enforces the schema the preprocessor was fitted on: any missing required column raises `ValueError`; extra columns (e.g. a new wind station appearing later) are silently dropped. This catches the failure mode where a held-out year's feature matrix doesn't line up with the training-time decisions — without the class, that mismatch would surface as a silently wrong prediction.

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
| `wave_eda.py` | Wave-only EDA across all eight buoys: coverage, distributions, seasonality, direction, autocorrelation, cross-source correlation. Saves eight `wave_*` PNGs to `notebooks/figures/`. |
| `lasso_ablation.py` | Per-source Lasso(α=0.001) ablation: trains on Mooloolaba alone, then plus each extra source (and the Gold-Coast + Palm-Beach pair) on the same 2015-2024 split. Prints a README-ready Markdown table. |
| `wind_eda.py` | Wind-only EDA across the available stations: coverage, time series, autocorrelation, direction roses, station comparison. Saves six `wind_*` PNGs to `notebooks/figures/`. |
| `wave_wind_eda.py` | Joint wave + wind EDA: alignment overview, feature-horizon screening, joint distributions. Saves three `wave_wind_*` PNGs to `notebooks/figures/`. |

## Results

All runs use a chronological 80/20 split on the **2015-2024 AEST** window (Brisbane time = UTC+10, no DST) — the span where the full-coverage wind stations have data. Skill score is vs. persistence on the same test split. The default source set is **5 neighbour buoys** (Brisbane, Caloundra, Gold Coast, North Moreton Bay, Tweed Heads) and **3 wind stations** (Mountain Creek, Deception Bay, Lytton); Palm Beach (2017-onwards) and Southport (mid-2018-onwards) are held out for the wider-set / shorter-window comparison below. The table below is a curated cross-section; find the full set of logged runs in `experiments.jsonl`.

| Model | Data sources | RMSE (cm) | Skill |
|-------|-------------|------|-------|
| Persistence (baseline) | Mooloolaba | 26.5 | — |
| LSTM (seq_len=48, hidden=64, 1 layer, 3 epochs, weight_decay=1e-4) | Mooloolaba + 5 neighbours + wind | 24.9 | +12.4% |
| RNN (seq_len=48, hidden=128, 2 layers, 3 epochs, weight_decay=1e-5) | Mooloolaba + 5 neighbours + wind | 23.9 | +18.8% |
| HGB (persistence-residual target) | Mooloolaba + 5 neighbours + wind | 23.4 | +22.6% |
| GRU (seq_len=48, hidden=64, 1 layer, 3 epochs, weight_decay=1e-4) | Mooloolaba + 5 neighbours + wind | 23.2 | +23.6% |
| TCN (seq_len=48, channels=(64,), 1 block, 2 epochs) | Mooloolaba + 5 neighbours + wind | 23.1 | +24.1% |
| Lasso (alpha=0.001) | Mooloolaba + 5 neighbours + wind | 23.1 | +24.1% |
| Ridge (alpha=1.0) | Mooloolaba + 5 neighbours + wind | 23.1 | +24.3% |
| **NanMean ensemble (Ridge + Lasso + HGB)** | Mooloolaba + 5 neighbours + wind | **22.7** | **+27.0%** |

**Notes:**

- **Best single model:** Ridge and TCN tie on RMSE. **Best overall:** nanmean ensemble of Ridge + Lasso + HGB (same family as previous best, ~0.4 skill points above the previous published 26.6%).
- The linear/tree rows are the best runs of `notebooks/linear_playground.py` on the 8-source feature set (5 neighbour buoys + 3 wind stations, ~315 features), with linear models on robust-scaled features (`fc.scale_features`).
- The sequence rows use the same best per-class configs from `notebooks/seq_sweep.py` that previously won on the smaller (4 nb + 2 wind) feature set. **TCN and GRU survive the larger input space; RNN and LSTM regress and want re-tuning** (e.g. wider hidden, more epochs) — a fresh sweep would close that gap.
- All runs go through a `fc.drop_sparse_columns(max_nan_frac=0.5)` step before imputation. This removed Lytton's `wind_speed_std_ms` (87.9 % NaN train-set) and its 12 lag/rolling derivatives — that column is essentially absent from the station's feed, and mean-imputation would silently turn it into a near-constant feature that confuses gradient-based seq models (TCN went from −1.7 % skill to +24.1 % skill once the dead channel was dropped).
- **Robust vs standard scaler** (in-forecaster, sequence models): robust helped the simpler recurrent nets — RNN +2.7, GRU +2.6 skill points — but was a wash for LSTM and TCN. Wave data's heavy tail favours a robust scale, but only where the model isn't already absorbing it.
- **Adam `weight_decay`** (1e-4 for the gated cells, 1e-5 for the vanilla RNN) enabled an extra training epoch without overfit (tuned on the smaller feature set; revisit alongside any seq sweep).
- Inter-layer `rnn_dropout` was tested but never beat the L=1 winners.

### Phase B: wider source set on a shorter window (2019-2024)

The default Results table holds the source set to what's available across the full 2015-2024 AEST span. The 2019-2024 AEST window unlocks three more sources — Palm Beach (deployed 2017), Southport wind (deployed mid-2018), and Wide Bay buoy (deployed 2019, the only buoy upstream of northerly swells). Same chronological 80/20 split as everywhere else; persistence baseline on this shorter window is 28.3 cm (vs 26.5 cm on the full window) because the recent years have more storm volatility.

| Model | Narrow (5nb + 3w) RMSE / Skill | Wide (7nb + 4w) RMSE / Skill | Δ skill |
|-------|--------------------------------|------------------------------|---------|
| Persistence | 28.30 cm / — | 28.30 cm / — | — |
| LSTM | 25.57 / +18.7% | 26.92 / +10.0% | −8.7 |
| RNN | 24.77 / +23.6% | 24.83 / +23.2% | −0.4 |
| GRU | 24.14 / +27.6% | 24.75 / +23.8% | −3.8 |
| HGB | 24.79 / +23.2% | 24.82 / +23.0% | −0.2 |
| TCN | 24.49 / +25.3% | 24.54 / +25.0% | −0.3 |
| Lasso | 24.42 / +25.6% | 24.33 / +26.2% | +0.6 |
| Ridge | 24.25 / +26.6% | 24.18 / +27.0% | +0.4 |
| **NanMean ensemble (Ridge + Lasso + HGB)** | **23.93 / +28.5%** | **23.90 / +28.7%** | **+0.2** |

Reading:

- **Adding Palm Beach + Southport + Wide Bay buys ~0.5 skill points for linear models and essentially nothing for the ensemble** — the wider source set is not a step change. The Gold Coast / Brisbane / Mooloolaba buoys already cover most of the explainable variance for a +12h forecast at this site.
- **Sequence models regress further with the wider set**, repeating the Phase A pattern: their published-best hyperparameters were tuned for the smaller feature matrix and don't generalise to ~38 input channels (38 vs 31 narrow vs 22 in the originally-published 4nb+2w runs). A fresh `seq_sweep.py` run on the wide set would close the gap.
- **Persistence is harder to beat on 2019-2024 by absolute RMSE but easier by skill** — the recent window is noisier, so everything has higher RMSE, but the persistence baseline is also higher, so the *gap* widens. The ~+28.7 % ensemble skill on Phase B looks better than +27.0 % on the headline table but reflects a different — and harder — test distribution.
- **Wide Bay's per-year coverage is uneven on this window** (33% in 2019, 32% in 2021, 61% in 2024 — see `notebooks/figures/wave_coverage.png`). It survives because mean-imputation fills the gaps with the buoy's mean, but the row-level sparsity is what limits the wider-set gain.

### Lasso: incremental value of each data source

To check that the extra sources actually carry signal, here is a plain `Lasso(alpha=0.001)` trained on the Mooloolaba buoy alone, then on Mooloolaba plus each extra source *in isolation* (not cumulative). The Gold-Coast-plus-Palm-Beach row is the one cumulative entry, since the two are close enough that an independent-vs-pair comparison is the question of interest. "Non-zero coefs" is how many of the feature columns Lasso kept, and "Top feature" is the largest `|coef|`. Reproduce with `./.venv/bin/python notebooks/lasso_ablation.py`.

| Data sources | RMSE (cm) | Skill | Non-zero coefs | Top feature |
|--------------|-----------|-------|----------------|-------------|
| Mooloolaba only | 25.3 | +9.2% | 42 / 107 | `hsig_m` |
| + Caloundra | 25.2 | +10.0% | 47 / 120 | `hsig_m` |
| + Brisbane | 24.2 | +17.0% | 52 / 120 | `brisbane_hsig_m` |
| + Gold Coast | 24.1 | +17.8% | 51 / 120 | `gold-coast_hsig_m` |
| + North Moreton Bay | 25.1 | +10.3% | 49 / 120 | `hsig_m` |
| + Tweed Heads | 25.2 | +9.8% | 52 / 120 | `hsig_m` |
| + Palm Beach | 25.2 | +14.4% | 53 / 120 | `hsig_m` |
| + Gold Coast + Palm Beach | 24.7 | +18.3% | 56 / 133 | `gold-coast_hsig_m` |
| + wind (4 stations) | 24.4 | +15.3% | 116 / 302 | `hsig_m` |

Every added source helps, but not equally:

- **Brisbane and Gold Coast** (the southern, swell-upstream buoys) are worth ~7-8 skill points on their own — Gold Coast even displaces the buoy's own `hsig_m` as the top feature.
- **Palm Beach** adds ~5 skill points alone — a meaningful gain, lower than Gold Coast (its near-neighbour) but well above Caloundra/NMB.
- **Tweed Heads** barely moves the needle on its own, despite a full 2015-2025 history. The zero-lag correlation with Mooloolaba is only ~0.65 (vs ~0.79 for Gold Coast and ~0.91 for Caloundra), so the southernmost buoy genuinely sits in a different swell context.
- **Gold Coast + Palm Beach together** lifts skill ~0.4 points over Gold Coast alone — they're partially redundant (the buoys are ~25 km apart) but Lasso keeps coefficients on both, so the two-source pair beats either alone. The current architecture handles this without any merging logic: each source's columns are prefixed, NaN-imputed, and offered to the model side by side.
- **Wind** with all four stations (now including Lytton and Southport) reaches +15.4%, up from +14.2% with the original two stations — the additional coverage at the Brisbane river mouth and Gold Coast helps modestly.

The ablation is single-source-vs-baseline, not cumulative; the full-source playground runs in the Results table above stack everything and outperform every row here.

## Model performance

Each new year is scored as a true blind set against three pre-committed models - best linear, best neural, best ensemble. The QLD wind 2025 release is the current blocker on the first row.

**Pre-committed candidates for 2025**, all trained on 2015-2024 with Mooloolaba + 5 neighbours + 3 wind stations:

- **Linear** — Ridge (alpha=1.0)
- **Neural** — TCN (seq_len=48, channels=(64,), 1 block, 2 epochs)
- **Ensemble** — NanMean of Ridge + Lasso + HGB

Scoring a new year against these committed candidates is a re-fit of the same recipe on the same training data, not a load of a serialised model. The `Preprocessor` fitted alongside each model captures the drop list, imputer means, and scaler stats, so the held-out year sees the same transformation the model was trained against — including any schema drift (extra columns are dropped, missing required columns raise). If you want to freeze the artifact instead of the recipe, `preproc.save(...)` + `pickle.dump(model, ...)` pair cleanly.

| Year | Model | RMSE (cm) | Skill |
|------|-------|-----------|-------|
| 2025 | Ridge | _TBD_ | _TBD_ |
| 2025 | TCN | _TBD_ | _TBD_ |
| 2025 | Ensemble | _TBD_ | _TBD_ |

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
│   │   ├── preprocess.py       # drop_sparse_columns, mean_impute, scale_features + Preprocessor (fit/transform/save/load)
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
├── pyproject.toml              # deps + forecast/viz extras (managed by uv)
└── uv.lock                     # pinned dependency graph (regen: uv lock)
```

**Package layout rationale.** The three packages are deliberately split so a trained forecaster can be imported without HTTP/CKAN deps, and `viz` stays decoupled from the models. All three live under `src/` via an editable install so scripts share the import path without `sys.path` hacks.

## Future Expansions

1. **Predict quantiles, not just the mean.** Quantile HGB (`HistGradientBoostingRegressor(loss="quantile", quantile=q)`) for P10/P50/P90 is a ~10-line addition; conformalised intervals over Ridge are similar. Also the structural fix for the confirmed tail underfit — residuals-vs-predicted plots show bias concentrated at high wave heights, a fat-tailed-residual-meets-MSE problem that a target transform can't fix. A quantile/pinball loss stops the model hedging the tail.

2. **Multi-horizon forecasts.** `HORIZON_STEPS` in `forecast/config.py:11` is centralised but the pipeline only fits one h. At 12h persistence is brutal (autocorr 0.81); at 24/48/72h it collapses, and that's where a real model has room to win. This is the change that would make the project read as a surf forecast model rather than a +12h `hsig` regressor.

3. **Multi-output forecasts.** A 2 m `hsig` from 90° at 14 s breaks very differently from 2 m from 150° at 8 s on the same beach. Forecast `tp_s` and `peak_dir_deg` jointly (`MultiOutputRegressor`, same API as item 2) so downstream code can run a break-specific transform.

4. **Partitioned swell.** Sea vs primary vs secondary swell components, if any QLD or BOM resource exposes them. Bimodal swells will never fit a single `hsig` number.

5. **Mooloolaba tide gauge.** The QLD portal hosts a tide gauge (`mooloolaba-tide-gauge-archived-interval-recordings`) directly at the target buoy location. Schema is trivial — `Date`, `Time`, `Reading` (water level in m) — and tidal range could carry second-order modulation of `hsig_m` that the wave/wind feeds miss. **The catch:** only 2023-2025 are in the CKAN Datastore API (`datastore_active: true`); pre-2023 years are flat CSV/TXT resources that the current `paginate_records` path does not handle. Cleanest split is a new `qld_ckan.tide` sub-package with a flat-resource downloader. Reasonable middle ground: wire tide for 2023-2025 only as a fast experiment, see whether skill moves at all, and only build the older-years ingestion if it does.

6. **Long-cadence historical bundles.** The CKAN catalogue also hosts deeper-history wave bundles for the swell-upstream buoys — Mooloolaba 2000-2014 (1-h cadence), Brisbane 1976-2011 (12-h), Gold Coast 1987-2014 (6-h), Tweed Heads 1995-2011 (1-h). These are intentionally excluded from `qld_ckan.wave.constants.BUOYS` because the project's pipeline assumes a strict 30-minute axis and the bundles' minute offsets drift (e.g. 08:55, 14:56, 20:53). Anyone wanting to use them for low-frequency climatology, storm-event sampling, or a multi-cadence experiment would need a parallel ingestion path: a separate cadence parameter on the wave pipeline, a snap-to-grid (floor + dedup) step before reindex, and a downstream join strategy for marrying coarse historical context with the 30-min modern grid. The CKAN resource IDs are documented in the buoy package pages — `coastal-data-system-waves-{slug}` on `data.qld.gov.au` — and Queensland's brief 1989-1992 DST window inside the older bundles forces the `Australia/Brisbane` localisation to either switch to fixed UTC+10 (`Etc/GMT-10`) or carry per-row DST handling, so that decision is part of the ingestion design too.