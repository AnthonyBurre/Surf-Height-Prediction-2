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

### Linear models

Nine configs were swept through `notebooks/linear_playground.py`, all scored on the same pinned test window (2023-01-01 → 2024-12-31 AEST) so RMSE is directly comparable. Each config runs Ridge (α=1), Lasso (α=0.001), HGB-on-residuals (`max_iter=800`, `lr=0.03`, `depth=6`), and a nanmean ensemble of the three. Persistence on the shared window is **RMSE 0.3996 m** (39.96 cm).

| Config | Window | Sources | Feats | Best member | Ensemble RMSE (m) | Skill vs persistence |
|---|---|---|---|---|---|---|
| **v2 wide** | 2019–2024 | 7 buoys + 4 wind | 406 | Lasso 0.3481 | **0.3436** | **+0.2610** |
| v6 ridgehi (α=10 / α=5e-4) | 2015–2024 | 5 buoys + 3 wind | 328 | HGB 0.3470 | 0.3439 | +0.2598 |
| v1 baseline | 2015–2024 | 5 buoys + 3 wind | 328 | HGB 0.3470 | 0.3441 | +0.2590 |
| v9 tweed + mc-wind | 2015–2024 | tweed-heads + mountain-creek wind | 172 | HGB 0.3470 | 0.3446 | +0.2567 |
| v3 no-wind | 2015–2024 | 5 buoys, no wind | 172 | HGB 0.3477 | 0.3446 | +0.2566 |
| v5 dense lags | 2015–2024 | 5 buoys + 3 wind | 433 | HGB 0.3488 | 0.3452 | +0.2542 |
| v7 HGB-heavy alone | 2015–2024 | 5 buoys + 3 wind | 328 | HGB 0.3469 | — | +0.2460 |
| v8 mc-wind only | 2015–2024 | mooloolaba + mountain-creek wind | 159 | HGB 0.3500 | 0.3475 | +0.2443 |
| v4 solo (no neighbours, no wind) | 2015–2024 | mooloolaba only | 107 | HGB 0.3527 | 0.3490 | +0.2379 |

The headline takeaway is that **every reasonable config lands inside a ~5 mm RMSE band** (0.3436–0.3490) — the model family, regularisation strength, and feature grid all matter less than which external sources are in the feature matrix. Concretely:

- **One upstream buoy carries almost all the neighbour signal.** v9 (Tweed Heads + Mountain Creek wind, 1 buoy + 1 wind) hits ensemble RMSE 0.3446 — bit-for-bit tied with v3 (5 buoys, no wind, also 0.3446) and within 1 mm of v1 (5 buoys + 3 wind, 0.3441). Tweed Heads is ~100 km south of Mooloolaba and sees southerly swells first; once you have it, Brisbane / Caloundra / Gold Coast / North Moreton Bay add nothing measurable.
- **The source ladder is steep then flat.** Mooloolaba alone (v4) 0.3490 → add Mountain Creek wind (v8) 0.3475 (−1.5 mm) → add Tweed Heads buoy (v9) 0.3446 (−2.9 mm) → add 4 more neighbour buoys + 2 more wind stations (v1) 0.3441 (−0.5 mm). The first two sources beyond the primary buoy do almost all the work.
- **Wide window helps the ensemble, marginally.** v2 (2019-2024, 7 buoys + 4 wind) edges out v1 by 0.5 mm, despite halving the training set — the two extra neighbour buoys (Palm Beach, Wide Bay) and Southport wind compensate.
- **The default feature grid is already near-optimal.** v5 doubles the lag/rolling/momentum density and gets *worse* by 1.1 mm — the marginal columns are noise the linear models then have to regularise away.
- **The ensemble is the right shipping artefact.** Every config's ensemble beats every individual member in it, even though the members are highly correlated. With three near-equally-good models there's no Bayesian-averaging dilemma: the nanmean costs nothing and shaves another 0.3–0.5 cm.
- **HGB-on-residuals is the strongest single model**, modestly ahead of Ridge and Lasso across configs. Direct-target HGB (run separately in v7-style trials) is consistently worse, which matches the priors — letting persistence handle the level and giving HGB only the delta is a meaningfully easier learning problem.

For full per-model rows including MAE and bias, see `experiments.jsonl` (filter on `name` starting with `lineopt_v`).

## Real world performance

Each new year that passes can be scored as a true blind set against our best models. This way we evaluate them against brand new data that wasn't implicitly leaked through the train/test iterative process. Awaiting the QLD wind 2025 release expected September 2026.

**Pre-committed candidates for 2025**:

- TBD
- TBD
- **TBD Ensemble**

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
| baselines       | `PersistenceForecaster`, `ClimatologyHourForecaster` |
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
| `wave_eda.py` | Wave-only EDA across all eight buoys: coverage, distributions, per-year target stats (Mooloolaba `hsig_m` boxplot + summary lines), seasonality, direction, autocorrelation, cross-source correlation. Saves nine `wave_*` PNGs to `notebooks/figures/`. |
| `baseline_diagnostics.py` | Fits Persistence and ClimatologyHour on pre-2023 Mooloolaba; scores on the same 2023-01-01 → 2024-12-31 window as the linear sweep. Saves `baseline_residuals.png` — two stacked residual time series (raw + 7-day rolling mean) plus an overlaid residual-density panel. |
| `wind_eda.py` | Wind-only EDA across the available stations: coverage, time series, autocorrelation, direction roses, station comparison. Saves six `wind_*` PNGs to `notebooks/figures/`. |
| `wave_wind_eda.py` | Joint wave + wind EDA: alignment overview, feature-horizon screening, joint distributions. Saves three `wave_wind_*` PNGs to `notebooks/figures/`. |
| `lasso_ablation.py` | Per-source Lasso(α=0.001) ablation: trains on Mooloolaba alone, then plus each extra source (and the Gold-Coast + Palm-Beach pair) on the same 2015-2024 split. Prints a README-ready Markdown table. |



## Future directions

1. **Predict quantiles, not just the mean.** Quantile HGB (`HistGradientBoostingRegressor(loss="quantile", quantile=q)`) for P10/P50/P90, or conformalised intervals over Ridge. Also the structural fix for the tail underfit — residuals-vs-predicted shows bias at high wave heights that pinball loss addresses and a target transform won't.

2. **Multi-horizon forecasts.** `HORIZON_STEPS` (`forecast/config.py:11`) is centralised but the pipeline only fits one h. At 24/48/72h persistence collapses, and that's where a real model has room to win. Makes the project read as a surf forecast rather than a +12h `hsig` regressor.

3. **Multi-output forecasts.** 2 m at 90°/14 s breaks very differently from 2 m at 150°/8 s. Forecast `tp_s` and `peak_dir_deg` jointly (`MultiOutputRegressor`) so downstream code can apply break-specific transforms.

4. **Mooloolaba tide gauge.** Resource `mooloolaba-tide-gauge-archived-interval-recordings` at the target buoy location — schema is `Date`, `Time`, `Reading` (water level m). Tidal range may carry second-order modulation of `hsig_m`. Catch: only 2023-2025 are on the CKAN Datastore API; pre-2023 are flat CSV/TXT resources not handled by `paginate_records`. Plan: new `qld_ckan.tide` sub-package with a flat-resource downloader. First wire 2023-2025 as a fast experiment; only build older-year ingestion if skill moves.

5. **Long-cadence historical bundles.** Deeper wave history for swell-upstream buoys: Mooloolaba 2000-2014 (1h), Brisbane 1976-2011 (12h), Gold Coast 1987-2014 (6h), Tweed Heads 1995-2011 (1h). Excluded from `qld_ckan.wave.constants.BUOYS` because the pipeline assumes a 30-min axis and these have drifting minute offsets (e.g. 08:55, 14:56). Needs: a cadence parameter on the wave pipeline, snap-to-grid (floor + dedup) before reindex, and a join strategy mixing coarse history with the 30-min grid. Resource IDs at `coastal-data-system-waves-{slug}` on `data.qld.gov.au`. QLD's 1989-1992 DST window forces choosing fixed UTC+10 (`Etc/GMT-10`) or per-row DST.