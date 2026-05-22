# Surf Height Prediction 2

An exercise in predictive modeling, this project is all about forecasting significant wave height (`hsig_m`) as measured by the Mooloolaba wave buoy off the sunny coast in Queensland, Australia.


## Objective

Given observations up to time *t* (30-minute cadence), predict `hsig_m` at *t + 12h*. 

At 12h the autocorrelation of `hsig_m` is ≈ 0.81, so persistence is a stiff baseline. Evaluation is a chronological 80/20 split; the headline metric is **skill score versus persistence** (a positive score means the model added information over "it'll be the same as now"). 

## Data sources

All data in this project comes from the [Queensland Government open data portal](https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation), which provides us with several wind and wave monitoring stations in the region.

Thankfully with raw AEST records we don't have to worry about time changes, so `pipeline.clean` can simply localize the naive AEST records to Australia/Brisbane and every unified CSV carries a gap-free `datetime` index in Brisbane time.

### A note on upstream revisions

The QLD portal publishes these as *derived, delayed-mode* wave parameters, and it periodically re-derives and republishes whole yearly resource files. Comparing an October 2025 snapshot against a re-download confirmed this: of ~178k shared timestamps, **26.9% changed**, with a clear signature rather than random drift. No revision notice is published, so the behaviour is documented here.

- **Whole records are recomputed, not just `hsig_m`.** `hmax_m`, `tz_s`, `tp_s`, and `peak_dir_deg` all change on the same ~27% of rows (`sst_c` on ~26%) — the buoy spectra were reprocessed, not patched.
- **The change is symmetric.** New values are higher 50.3% / lower 49.7% of the time (mean Δ ≈ 0, −0.011 m) and the maximum is unchanged (5.204 m), so it is not a clipping, units, or one-sided shift — but magnitudes are large (median |Δ| = 0.42 m).
- **Revisions are temporally clustered, not periodic.** They form 195 contiguous blocks (median ~5 days, max 18), never scattered single points, with no time-of-day pattern.
- **They concentrate in big seas.** Waves >3 m were revised 40% of the time (mean |Δ| 0.63 m) vs ~25% / 0.11 m for 0.5–1.5 m waves, and the largest blocks all fall in the Dec–Mar storm/cyclone season (e.g. 2025-01-27→02-15, 2023-12-01→12-11).
- **Three years are untouched.** 2017, 2018, and 2021 are byte-identical; 2015/16/19/20/22/23/24/25 were republished.

The net effect is that the revised data is **rougher at every lag** (12h autocorrelation dropped 0.85 → 0.74), which makes the +12h persistence baseline harder to beat: persistence RMSE rose from 26.5 cm on the old snapshot to ~40 cm now. The revision is a data-quality improvement (more accurate storm-period measurements), not a regression — but it means **absolute RMSE is not comparable across snapshots**, while skill-vs-persistence is largely preserved. `test_persistence_baseline_matches_documented_values` pins the current baseline so a future revision is caught rather than silently shifting the headline numbers.



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

![Wave coverage](notebooks/figures/wave_coverage.png)

### Wind (air-quality monitoring network)

Hourly cadence, 10 m ultrasonic wind sensors on the QLD air-quality monitoring stations. Mountain Creek pairs with the Mooloolaba buoy, Deception Bay sits ~50 km south on Moreton Bay, Lytton is at the mouth of the Brisbane River (paired with the Brisbane buoy), and Southport sits on the Gold Coast (paired with the Gold Coast / Palm Beach buoys). Pollutant and temperature fields are dropped at clean time, leaving:

| Column | Description |
|--------|-------------|
| `wind_dir_deg` | Wind direction (degrees true north) |
| `wind_speed_ms` | Wind speed (meters/second) |
| `wind_sigma_theta_deg` | Wind direction standard deviation (degrees) |
| `wind_speed_std_ms` | Wind speed standard deviation (meters/second) |

The wind frame is reindexed onto the 30-minute wave grid by forward-fill.

![Wind coverage](notebooks/figures/wind_coverage.png)

### Dataset selection: length vs breadth

The coverage grids define the trade space for any experiment. The choice is a breadth-vs-depth call across two axes — how far back to train, and how many neighbour sources to include:

1. **Standard window, full neighbour set (2015-2024 AEST, 5 buoys + 3 wind stations).** The default for the headline results. Lytton wind starts 2015, so this window is the deepest one where every "always-on" source has data. Mooloolaba + Brisbane / Caloundra / Gold Coast / North Moreton Bay / Tweed Heads neighbours and Mountain Creek / Deception Bay / Lytton wind — 10 years of training, no late-deployment gaps.
2. **Short window, wide set (2019-2024 AEST, 7 buoys + 4 wind stations).** Palm Beach (deployed 2017), Southport wind (mid-2018), and Wide Bay (2019, the only buoy upstream of northerly swells) only join here. Six years of training in exchange for two extra neighbour buoys and one extra wind station.

## Data preparation

### Feature engineering

The feature matrix is assembled in three layers, each a single call:

1. **Base primary-buoy matrix** (`fc.build_buoy_features`) — circular encoding, hour/doy time features, lags, rolling stats, momentum. Lag/rolling/delta grids are tunable via `FeatureConfig`. For sequence models, `fc.build_seq_features` swaps in raw channels with no pre-built lags (the model windows its own input).
2. **Neighbour buoys** (`fc.add_neighbour_features`) — raw value, lag copies, and rolling mean/std per neighbour column, reusing the same `FeatureConfig`.
3. **Wind stations** — same `add_neighbour_features` call on each wind station's columns. `fc.load_wind` sin/cos-encodes `wind_dir_deg` and station-prefixes every column so cross-station features stay distinguishable.

### Preprocessing pipeline

`forecast.preprocess.Preprocessor` bundles the three steps the playgrounds run between `chronological_split` and `model.fit`:

1. **Drop sparse columns** — any column whose **train-set** NaN fraction exceeds `max_nan_frac` (default 0.5) is removed. Mean-imputing a near-empty column gives a near-constant feature that silently corrodes gradient-based sequence models (see the dropped-column note under Results for a concrete TCN example). The `wave_column_coverage.png` and `wind_column_coverage.png` EDA figures surface candidates ahead of time.
2. **Mean impute** — column-wise mean from training data fills remaining NaNs.
3. **Scale** (optional) — `"robust"` (median/IQR) or `"standard"`. Linear models default to robust because wave data is heavy-tailed and storm spikes would inflate a standard-deviation scale. Trees (HGB) take the raw matrix. Sequence models scale internally (`scaler="robust"` or `"standard"`, fit on train) and consume the unscaled `build_seq_features` frame. `*_sin`/`*_cos` columns pass through untouched in all cases.

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


## Model selection and tuning

All runs use a chronological 80/20 split on the **standard window** (2015-2024 AEST, 5 neighbour buoys + 3 wind stations — see *Dataset selection* above). Brisbane time is UTC+10 year-round, no DST. Skill score is vs. persistence on the same test split. The table below is a curated cross-section; find the full set of logged runs in `experiments.jsonl`.

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

- **Best single model:** Ridge and TCN tie on RMSE. **Best overall:** nanmean ensemble of Ridge + Lasso + HGB.
- The linear/tree rows are the best runs of `notebooks/linear_playground.py` on the 8-source feature set (5 neighbour buoys + 3 wind stations, ~315 features), with linear models on robust-scaled features (`fc.scale_features`).
- The sequence rows use the same best per-class configs from `notebooks/seq_sweep.py` that previously won on the smaller (4 nb + 2 wind) feature set. **TCN and GRU survive the larger input space; RNN and LSTM regress and want re-tuning** (e.g. wider hidden, more epochs) — a fresh sweep would close that gap.
- All runs go through a `fc.drop_sparse_columns(max_nan_frac=0.5)` step before imputation. This removed Lytton's `wind_speed_std_ms` (87.9 % NaN train-set) and its 12 lag/rolling derivatives — that column is essentially absent from the station's feed, and mean-imputation would silently turn it into a near-constant feature that confuses gradient-based seq models (TCN went from −1.7 % skill to +24.1 % skill once the dead channel was dropped).
- **Robust vs standard scaler** (in-forecaster, sequence models): robust helped the simpler recurrent nets — RNN +2.7, GRU +2.6 skill points — but was a wash for LSTM and TCN. The heavy-tail benefit only shows up where the model isn't already absorbing it.
- **Adam `weight_decay`** (1e-4 for the gated cells, 1e-5 for the vanilla RNN) enabled an extra training epoch without overfit (tuned on the smaller feature set; revisit alongside any seq sweep).
- Inter-layer `rnn_dropout` was tested but never beat the L=1 winners.

### Wider source set, shorter training history (2019-2024)

Switching to the wide-set window (see *Dataset selection*) unlocks Palm Beach, Southport wind, and Wide Bay. Both feature sets are scored on the **same test slice** — narrow's natural 80/20 split timestamp (2023-01-01 09:30 → 2024-12-31 23:30, n=35,069) is reused as the wide-set cutoff via `seq_sweep.py` and the `test_start` knob in `linear_playground.py`, so the only differences between rows are feature set (5nb+3w vs 7nb+4w) and training-history depth (140k rows from 2015 vs 70k from 2019). Sequence-model hyperparameters were re-tuned per set (44 wide configs, 41 narrow); linear/HGB/ensemble rows are the best `linear_playground.py` runs.

| Model | Narrow (5nb + 3w) RMSE / Skill | Wide (7nb + 4w) RMSE / Skill | Δ skill |
|-------|--------------------------------|------------------------------|---------|
| Persistence | 39.96 cm / — | 39.96 cm / — | — |
| TCN | 38.58 / +6.9% | 37.67 / +11.3% | +4.4 |
| LSTM | 37.45 / +12.3% | 38.08 / +9.4% | −2.9 |
| Lasso | 35.15 / +22.7% | 34.82 / +24.2% | +1.5 |
| GRU | 34.96 / +23.6% | 35.11 / +23.0% | −0.6 |
| Ridge | 34.84 / +24.0% | 34.89 / +23.8% | −0.2 |
| HGB | 34.73 / +24.5% | 35.60 / +20.7% | −3.8 |
| RNN | 34.57 / +25.3% | 35.28 / +22.2% | −3.1 |
| **NanMean ensemble (Ridge + Lasso + HGB)** | **34.42 / +25.9%** | **34.40 / +26.0%** | **+0.1** |

Best wide configs from the sweep: RNN `sl48 h256 L2 ep3 wd=1e-4 do=0.1`, GRU `sl48 h64 L1 ep2`, LSTM `sl48 h128 L2 ep3 wd=1e-4 do=0.1`, TCN `sl48 channels=(128,)×4 ep3 wd=1e-4 do=0.2`. Best narrow: RNN `sl48 h128 L2 ep3 wd=1e-4`, GRU `sl48 h64 L1 ep2`, LSTM `sl48 h128 L1 ep3 wd=1e-4`, TCN `sl48 channels=(64,) ep2 do=0.1`.

Reading:

- **The wider source set is essentially a wash on a fair window.** The ensemble is dead-flat (+0.1 skill points), Ridge / GRU / Lasso move by ≤1.5 points, and the modest gains (Lasso, TCN) and losses (HGB, RNN) roughly cancel. Mooloolaba + Gold Coast + Brisbane already capture most of the explainable +12h variance at this site; Palm Beach + Southport + Wide Bay don't add much that survives regularisation or ensembling.
- **TCN is the lone material winner on wide** (+4.4 skill points). A deeper stack (4 blocks at 128 channels) finally finds a config that exploits the wider input — though TCN still trails the linear winners by ~14 skill points and isn't competitive yet on this task.
- **HGB and RNN regress ~3 skill points on wide.** Both lose more from halving the training history (8y → 4y) than they gain from the extra sources. HGB drops from top-2 single model on narrow to mid-pack on wide.
- **The best single seq model on each set is now within striking distance of the ensemble** — narrow RNN +25.3 vs ensemble +25.9; wide GRU +23.0 vs ensemble +26.0. The previous headline note that RNN/LSTM "want re-tuning" was right; the broadened-grid sweep absorbs most of the gap.
- **LSTM trails the family on both sets** (+12% / +9%) — same hidden / layers / epoch envelope as RNN/GRU, but the gated recurrence isn't earning its parameters here.
- **Wide Bay's per-year coverage is uneven on this window** (33% in 2019, 32% in 2021, 61% in 2024 — see `notebooks/figures/wave_coverage.png`). It survives because mean-imputation fills the gaps with the buoy's mean, but the row-level sparsity remains a known cap on the wider-set gain.

Bottom line: at parity on the same test window, **the narrow set is the better deployment choice** — same headline skill, twice the training data, fewer late-deployment coverage gaps.

### Lasso: incremental value of each data source

To check that the extra sources actually carry signal, here is a plain `Lasso(alpha=0.001)` trained on the Mooloolaba buoy alone, then on Mooloolaba plus each extra source *in isolation* (not cumulative). The Gold-Coast-plus-Palm-Beach row is the one cumulative entry, since the two are close enough that an independent-vs-pair comparison is the question of interest. "Non-zero coefs" is how many of the feature columns Lasso kept, and "Top feature" is the largest `|coef|`. Reproduce with `./.venv/bin/python notebooks/lasso_ablation.py`.

| Data sources | RMSE (cm) | Skill | Non-zero coefs | Top feature |
|--------------|-----------|-------|----------------|-------------|
| Mooloolaba only | 35.7 | +20.5% | 47 / 107 | `hsig_m` |
| + Caloundra | 35.6 | +20.7% | 55 / 120 | `hsig_m` |
| + Brisbane | 35.3 | +21.8% | 57 / 120 | `hsig_m` |
| + Gold Coast | 35.3 | +22.1% | 56 / 120 | `hsig_m` |
| + North Moreton Bay | 35.6 | +20.7% | 54 / 120 | `hsig_m` |
| + Tweed Heads | 35.4 | +21.8% | 53 / 120 | `hsig_m` |
| + Palm Beach | 36.3 | +20.8% | 53 / 120 | `hsig_m` |
| + Gold Coast + Palm Beach | 36.1 | +21.4% | 61 / 133 | `hsig_m` |
| + wind (4 stations) | 35.4 | +21.5% | 113 / 302 | `hsig_m` |

The picture is more compressed than it used to be — every extra source moves skill by under 2 points, and a few don't move it at all:

- **Brisbane and Gold Coast** (the southern, swell-upstream buoys) are still the strongest standalone additions, worth ~1.3-1.6 skill points each. Unlike before, neither displaces the buoy's own `hsig_m` as the top feature on the republished data.
- **Tweed Heads** is now competitive with Brisbane (+1.3 points), despite a zero-lag correlation with Mooloolaba of only ~0.65 — it carries real southern-swell signal that the regulariser keeps.
- **Caloundra and North Moreton Bay** barely move the needle (+0.2 points each), consistent with their near-neighbour status and limited unique signal.
- **Palm Beach** adds ~0.3 points alone, and the **Gold-Coast + Palm-Beach pair is actually worse than Gold Coast alone** (+21.4% vs +22.1%). The two buoys are ~25 km apart and Lasso keeps coefficients on both, but the extra columns appear to cost more in noise than they buy in signal at this regularisation strength.
- **Wind** with all four stations reaches +21.5%, ~1 point over Mooloolaba alone — modest, and below Gold Coast alone.

`hsig_m` is the top feature in every row, so on the republished data the neighbour buoys are useful *additions* rather than replacements for the buoy's own recent history.

## Real world performance

Each new year that passes can be scored as a true blind set against our best models. This way we evaluate them against brand new data that wasn't implicitly leaked through the train/test iterative process. Awaiting the QLD wind 2025 release expected September 2026.

**Pre-committed candidates for 2025**, all trained on 2015-2024 with Mooloolaba + 5 neighbours + 3 wind stations:

- **Linear** — Ridge (alpha=1.0)
- **Neural** — TCN (seq_len=48, channels=(64,), 1 block, 2 epochs)
- **Ensemble** — NanMean of Ridge + Lasso + HGB

Scoring a new year against these committed candidates is a re-fit of the same recipe on the same training data, not a load of a serialised model. The `Preprocessor` fitted alongside each model captures the drop list, imputer means, and scaler stats, so the held-out year sees the same transformation the model was trained against — including any schema drift (extra columns are dropped, missing required columns raise).

| Year | Model | RMSE (cm) | Skill |
|------|-------|-----------|-------|
| 2025 | Ridge | _TBD_ | _TBD_ |
| 2025 | TCN | _TBD_ | _TBD_ |
| 2025 | Ensemble | _TBD_ | _TBD_ |


## Reproducibility

### Setup

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --all-extras
```

Run tests after changes:

```bash
./.venv/bin/pytest src/tests/ -v
```

### Packages

- **`qld_ckan`**: Downloads yearly records from the QLD CKAN Datastore API, unifies the schema, and writes a cleaned CSV per source.
    - `qld_ckan.wave` (wave buoys) 
    - `qld_ckan.wind` (air-quality-station 10 m wind)
- **`viz`**: Source-agnostic plotting, organised by pipeline stage: shared time-series primitives, post-download EDA heatmaps, and post-modelling diagnostics.
- **`forecast`**: Target construction, chronological splits, feature engineering, baselines, metrics, and an evaluation harness. See *Available forecasters* below for the model list.

Experiment scripts in `notebooks/` run on top of these packages.

### Running the pipeline

Generate `data/` with these commands (one CSV per source):

```bash
# Wave - default Mooloolaba 2015-2025
./.venv/bin/python -m qld_ckan wave [--buoy brisbane|caloundra|gold-coast|north-moreton-bay|palm-beach|tweed-heads|wide-bay]

# Wind - default Mountain Creek 2010-2024
./.venv/bin/python -m qld_ckan wind [--station deception-bay|lytton|southport]
```

Both subcommands accept `--year-min` / `--year-max` (inclusive) to clip the registry before download.

```bash
./.venv/bin/python -m qld_ckan wave --buoy brisbane --year-min 2018 --year-max 2020
```

`forecast` exposes a flat import surface (`import forecast as fc`): target construction (`make_target` shifts `hsig_m` 24 steps ahead), chronological 80/20 split, feature builders, and an `evaluate_and_log` harness that scores `MAE / RMSE / Bias / SkillVsBaseline` against persistence and appends to `experiments.jsonl`. A typical call:

```python
result = fc.evaluate_and_log(
    fc.Ridge(alpha=1.0), X_tr, y_tr, X_te, y_te,
    name="ridge", data_sources=["mooloolaba"],
)
print(result.metrics)
```

### Available forecasters

| family          | classes                                                                         |
|-----------------|---------------------------------------------------------------------------------|
| baselines       | `PersistenceForecaster`, `SeasonalNaiveForecaster`, `ClimatologyHourForecaster` |
| linear / tree   | any scikit-learn regressor (Ridge, Lasso, HGB, …)                               |
| sequence models | `SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`       |

### Logging experiments

`fc.evaluate_and_log(...)` is a drop-in for `fc.evaluate(...)` that appends a record to `experiments.jsonl` (committed at repo root); `fc.log_run(result, ...)` covers results computed outside the harness. Read the log back as a DataFrame with `fc.read_log()`.

### Experiment scripts

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
| `wind_eda.py` | Wind-only EDA across the available stations: coverage, time series, autocorrelation, direction roses, station comparison. Saves six `wind_*` PNGs to `notebooks/figures/`. |
| `wave_wind_eda.py` | Joint wave + wind EDA: alignment overview, feature-horizon screening, joint distributions. Saves three `wave_wind_*` PNGs to `notebooks/figures/`. |
| `lasso_ablation.py` | Per-source Lasso(α=0.001) ablation: trains on Mooloolaba alone, then plus each extra source (and the Gold-Coast + Palm-Beach pair) on the same 2015-2024 split. Prints a README-ready Markdown table. |



## Future directions

1. **Predict quantiles, not just the mean.** Quantile HGB (`HistGradientBoostingRegressor(loss="quantile", quantile=q)`) for P10/P50/P90, or conformalised intervals over Ridge. Also the structural fix for the tail underfit — residuals-vs-predicted shows bias at high wave heights that pinball loss addresses and a target transform won't.

2. **Multi-horizon forecasts.** `HORIZON_STEPS` (`forecast/config.py:11`) is centralised but the pipeline only fits one h. At 24/48/72h persistence collapses, and that's where a real model has room to win. Makes the project read as a surf forecast rather than a +12h `hsig` regressor.

3. **Multi-output forecasts.** 2 m at 90°/14 s breaks very differently from 2 m at 150°/8 s. Forecast `tp_s` and `peak_dir_deg` jointly (`MultiOutputRegressor`) so downstream code can apply break-specific transforms.

4. **Mooloolaba tide gauge.** Resource `mooloolaba-tide-gauge-archived-interval-recordings` at the target buoy location — schema is `Date`, `Time`, `Reading` (water level m). Tidal range may carry second-order modulation of `hsig_m`. Catch: only 2023-2025 are on the CKAN Datastore API; pre-2023 are flat CSV/TXT resources not handled by `paginate_records`. Plan: new `qld_ckan.tide` sub-package with a flat-resource downloader. First wire 2023-2025 as a fast experiment; only build older-year ingestion if skill moves.

5. **Long-cadence historical bundles.** Deeper wave history for swell-upstream buoys: Mooloolaba 2000-2014 (1h), Brisbane 1976-2011 (12h), Gold Coast 1987-2014 (6h), Tweed Heads 1995-2011 (1h). Excluded from `qld_ckan.wave.constants.BUOYS` because the pipeline assumes a 30-min axis and these have drifting minute offsets (e.g. 08:55, 14:56). Needs: a cadence parameter on the wave pipeline, snap-to-grid (floor + dedup) before reindex, and a join strategy mixing coarse history with the 30-min grid. Resource IDs at `coastal-data-system-waves-{slug}` on `data.qld.gov.au`. QLD's 1989-1992 DST window forces choosing fixed UTC+10 (`Etc/GMT-10`) or per-row DST.