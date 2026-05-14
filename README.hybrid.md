# Surf Height Prediction 2

Forecasts significant wave height (`hsig_m`) 12 hours ahead at the Mooloolaba wave buoy, Queensland, using ten years of open-data buoy and AWS-wind observations (2015–2025). On a chronological 80/20 split, a Ridge regression conditioned on three neighbour buoys reaches **RMSE 0.244 m / +19.5 % skill** versus persistence on the 2024–2025 window. At this horizon, linear models with spatial neighbour features beat single-buoy recurrent networks (LSTM, GRU, TCN).

Three installed Python packages form the project:

- **`qld_ckan`** — ETL. Downloads yearly records from the QLD Government CKAN Datastore API, unifies schema, and writes a cleaned CSV on a per-source grid. Two sub-packages: `qld_ckan.wave` (wave-buoy network, 30-minute grid) and `qld_ckan.wind` (AWS station 10 m wind, hourly grid). Shared transport (retrying session, paginated GET, 404-skip year loop, `unify_frames`) lives at the umbrella level.
- **`viz`** — source-agnostic plotting, organised by pipeline stage. Shared time-series primitives, post-download EDA heatmaps, and post-modeling diagnostics.
- **`forecast`** — modelling. Target construction, chronological splits, feature engineering, baselines, metrics, an evaluation harness, and sequence-model forecasters (RNN / GRU / LSTM / TCN) built on PyTorch.

Experiment scripts in `notebooks/` run on top of these packages.

## Motivation

Operational surf and marine-safety forecasts care about lead times of hours-to-days. Twelve hours covers the planning window for surfers, lifeguards, and coastal works. It also sits inside the regime where the simplest possible model — *"it'll be the same as it is now"* — is already very strong: the 12-h autocorrelation of `hsig_m` at Mooloolaba is ≈ 0.81, so a useful model has to add information not already encoded in the most recent observation.

Queensland's open data portal exposes a chain of buoys along the south-east coast. The buoys south of Mooloolaba (Brisbane, North Moreton Bay) see southerly swells first; Gold Coast sees easterly events first. This neighbour-as-leading-indicator structure is what the best model in this repository exploits.

Operational regional forecasts use spectral wave models (NOAA WaveWatch III; SWAN). Commercial surf services blend such hindcasts with proprietary observations. The model here is much simpler — single-point regression on a chronological split — but is built end-to-end on free data and reports skill against a strict persistence baseline.

## Problem

Given buoy observations up to time *t* (30-minute cadence), predict `hsig_m` at *t + 12h* — 24 steps ahead. Evaluation is a chronological 80/20 split; the headline metric is **skill score versus persistence** (a positive score means the model added information over "it'll be the same as now").

At 12h the autocorrelation of `hsig_m` is ≈ 0.81, so persistence is a stiff baseline.

![12-h autocorrelation of `hsig_m` ≈ 0.81](notebooks/figures/wave_autocorrelation.png)

## Setup

Requires Python 3.14. Create a venv and install all pinned dependencies (including the editable local packages):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `data/` directory is gitignored — populate it by running the pipeline.

## Running the pipeline

`qld_ckan` exposes one CLI with two subcommands — `wave` for the buoy network, `wind` for the AWS stations. Each writes a cleaned CSV to `data/`.

```bash
# Default: Mooloolaba 2015–2025 → data/mooloolaba_wave_data_2015-2025.csv
./.venv/bin/python -m qld_ckan wave

# Any supported buoy
./.venv/bin/python -m qld_ckan wave --buoy brisbane
./.venv/bin/python -m qld_ckan wave --buoy caloundra
```

Supported buoys: `mooloolaba`, `brisbane`, `caloundra`, `gold-coast`, `north-moreton-bay`.

```bash
# Default: Mountain Creek 2015–2024 → data/mountain-creek_wind_data_2015-2024.csv
./.venv/bin/python -m qld_ckan wind

# Any supported station
./.venv/bin/python -m qld_ckan wind --station mountain-creek
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

`fc.build_buoy_features(df)` produces the full primary-buoy feature matrix (lags `[1, 2, 6, 24]`, rolling mean/std over `[12, 48]` shifted past-only, momentum deltas `[6, 12, 24, 48]`, sin/cos circular encoding of `peak_dir_deg`, plus hour-of-day and day-of-year time features) for any QLD wave buoy. Neighbour buoys are appended with `fc.add_neighbour_features(X, source_df, columns)`. Both accept a `FeatureConfig` to tune lag steps, rolling windows, and delta steps:

```python
cfg = fc.FeatureConfig(lag_steps=[1, 2, 6, 24], roll_windows=[12, 48])
X = fc.build_buoy_features(df, config=cfg)
```

For sequence models (LSTM / GRU / TCN), use `fc.build_seq_features(df)` — circular encoding and time features only, no pre-built lags (the model windows its own input).

### Available forecasters

| family          | classes                                                                         |
|-----------------|---------------------------------------------------------------------------------|
| baselines       | `PersistenceForecaster`, `SeasonalNaiveForecaster`, `ClimatologyHourForecaster` |
| linear / tree   | any scikit-learn regressor (Ridge, Lasso, HGB, …)                               |
| sequence models | `SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`       |

Skill is always reported vs. persistence; seasonal-naive and climatology-by-hour exist as diagnostic floors.

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

All runs use a chronological 80/20 split. Skill is vs. persistence on the same test window. Persistence baselines differ across windows (the full-history split, the 2024–2025 neighbour-buoy overlap, and the 2015–2024 wind-overlap window) so RMSEs are only directly comparable within the same window.

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
| Ridge | Mooloolaba + 3 neighbours | 2024-2025 | **0.244** | **+19.5%** |
| HGB | Mooloolaba + 3 neighbours | 2024-2025 | 0.258 | +10.0% |

The full set of logged runs is in `experiments.jsonl`.

![Neighbour buoys as leading indicators](notebooks/figures/wave_neighbour_predictive.png)

## Discussion

**Headline.** Ridge with three neighbour buoys reaches RMSE 0.244 m on the 2024–2025 window — a 19.5% skill improvement over persistence (RMSE 0.272 m). For a target whose 12-h autocorrelation is 0.81, this is a meaningful margin.

**Why neighbours help.** Brisbane and North Moreton Bay sit south of Mooloolaba and respond first to southerly swells; Gold Coast responds first to easterly events. The Ridge coefficient on a neighbour's recent `hsig_m` is doing approximately what a swell-propagation model would do analytically.

**Why sequence models lose at h = 12.** The LSTM/GRU/TCN models tested on the same windows do not beat persistence (the LSTM in the table sits at −55% skill). Two factors: (a) the engineered feature matrix already pre-computes the dominant lags that an RNN would otherwise have to discover, and (b) the spatial neighbour information that drives the best linear result is not reflected in single-buoy sequence inputs. Recent residual-training experiments (`notebooks/seq_playground.py`, May 2026) attempt to close this gap by training sequence models on persistence residuals rather than raw `hsig_m`.

**Where the model fails.** The current best Ridge run carries a small positive bias of ≈ +0.018 m, consistent with regression-to-the-mean on a long-tailed target — large wave events are rare in training, so squared-error optimisation hedges toward the bulk. A log- or arcsinh-transform of the target is the standard fix.

## Future work

The work in this repository is a baseline. The most promising directions:

**Evaluation rigor.** The current single 80/20 split means the "test set" has been implicitly used for model selection across the 55 logged runs. Hold out the last 6 months as a true blind set, OR move to expanding-window cross-validation (sklearn `TimeSeriesSplit`) for hyperparameter selection. Report per-condition error breakdowns (by sea state, season, direction) instead of a single aggregate.

**Modelling extensions.**

- **Uncertainty.** Quantile HGB (`HistGradientBoostingRegressor(loss="quantile", quantile=q)`) for P10/P50/P90, or conformalised intervals over Ridge.
- **Target transform.** Fit `np.log1p(hsig_m)` or `np.arcsinh(hsig_m / H₀)` and exponentiate at predict time to address high-tail under-prediction.
- **Physics features.** Wave power proxy `hsig_m² · tp_s`; swell × wind alignment `wind_speed · cos(wind_dir − peak_dir)`; `tp_s / wind_speed` as a wind-sea vs. groundswell separator. Multiplicative interactions that linear models cannot recover on their own.
- **Multi-horizon.** Direct forecasters at h ∈ {12, 24, 48, 72} via `MultiOutputRegressor`. At h = 12 persistence is brutal; at h ≥ 24 it collapses and a model genuinely earns its keep.

**Scope.** `hsig_m` is not surf height. A 2 m hsig from 90° at 14 s breaks very differently from a 2 m hsig from 150° at 8 s on the same beach. Joint forecast of `tp_s` and `peak_dir_deg` (multi-output) would let downstream code apply a break-specific transform. Integrating NOAA WaveWatch III hindcast (free, global, ~25 km grid) would bring partitioned swell (sea / primary / secondary) — bimodal swells will never fit a single-`hsig_m` regression.

## Running tests

```bash
./.venv/bin/pytest src/tests/ -v
```

Network calls are mocked, so tests run offline.

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

## Data

Queensland Government open data portal. Fetched via the CKAN Datastore API (`datastore_search`) rather than raw CSV downloads, so resource IDs remain stable across portal file renames: <https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation>.

**Wave buoys** (30-min grid). Mooloolaba (target, 2015–2025), Brisbane (2015–2025), North Moreton Bay (2010–2025), Caloundra (2013–2025), Gold Coast (2015–2025).

![Wave-buoy data coverage by year](notebooks/figures/wave_coverage.png)

The unified buoy CSV has a `datetime_utc` index at 30-minute intervals (raw records are AEST; `pipeline.clean` localises then converts to UTC):

| Column | Description |
|--------|-------------|
| `hsig_m` | Significant wave height (metres) — **target** |
| `hmax_m` | Maximum wave height (metres) |
| `tz_s` | Zero-crossing period (seconds) |
| `tp_s` | Peak period (seconds) |
| `peak_dir_deg` | Peak wave direction (degrees, circular) |
| `sst_c` | Sea surface temperature (°C) |

Missing or erroneous readings (`-99.9` in raw files) are replaced with `NaN` and the index is resampled onto a gap-free 30-minute grid.

**Air-quality / meteorology AWS network** (hourly grid, forward-filled to the 30-min wave grid). Mountain Creek (Sunshine Coast Council AWS at -26.69, 153.10 — effectively co-located with the Mooloolaba wave buoy) and Deception Bay (Moreton Bay, ~50 km south). Both carry the same 10 m wind schema (`wind_speed_ms`, `wind_dir_deg`, two dispersion stats); pollutant fields are dropped at clean time. Wind direction is sin/cos-encoded before being passed to `add_neighbour_features` — same pattern as the wave-buoy `peak_dir_deg` encoding in `build_buoy_features`. The forward-fill is strictly past-only (the 14:30 slot gets the 14:00 wind value).

The wave history is sliced to 2015–2024 to match the wind window — a separate persistence baseline is computed on that same window so skill scores are directly comparable to the wind-augmented runs.

## References

- Booij, N., Ris, R. C., & Holthuijsen, L. H. (1999). *A third-generation wave model for coastal regions: 1. Model description and validation.* J. Geophys. Res. Oceans 104(C4), 7649–7666.
- Tolman, H. L. (1991). *A third-generation model for wind waves on slowly varying, unsteady, and inhomogeneous depths and currents.* J. Phys. Oceanogr. 21(6), 782–797. (Foundational reference for NOAA WaveWatch III.)
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
- Queensland Government open-data portal: <https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation>.

## How to cite

```bibtex
@misc{surf_height_prediction_2,
  author       = {Anthony},
  title        = {Surf Height Prediction 2: Open-data buoy forecasting for the Queensland coast},
  year         = {2026},
  howpublished = {\url{https://github.com/<owner>/Surf-Height-Prediction-2}},
}
```

## License

See [LICENSE](LICENSE).
