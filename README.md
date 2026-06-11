# Surf Height Prediction 2

An exercise in predictive modeling, this project is all about forecasting significant wave height (`hsig_m`) as measured by the Mooloolaba wave buoy off the sunny coast in Queensland, Australia.


## Objective

Given observations up to time *t* (30-minute cadence), predict `hsig_m` at *t + 6h, 12h, 24h, 36h, 48h, 72h*.


## Data source

All data in this project comes from the [Queensland Government open data portal](https://www.data.qld.gov.au/organization/environment-tourism-science-and-innovation), which provides us with several wind and wave monitoring stations in the region. Since all raw records are AEST we don't have to worry about time changes, and every output unified CSV carries a gap-free Brisbane `datetime` index.

### Upstream revisions

The QLD portal publishes these as *derived, delayed-mode* wave parameters, and it periodically re-derives and republishes whole yearly resource files. Comparing an October 2025 snapshot against a re-download confirmed this: of ~178k shared timestamps, **26.9% changed**, with a clear signature rather than random drift. No revision notice is published, so the behaviour is documented here.

- **Whole records are recomputed, not just `hsig_m`.** `hmax_m`, `tz_s`, `tp_s`, and `peak_dir_deg` all change on the same ~27% of rows (`sst_c` on ~26%) — the buoy spectra were reprocessed, not patched.
- **The change is symmetric.** New values are higher 50.3% / lower 49.7% of the time (mean Δ ≈ 0, -0.011 m) and the maximum is unchanged (5.204 m), so it is not a clipping, units, or one-sided shift — but magnitudes are large (median |Δ| = 0.42 m).
- **Revisions are temporally clustered, not periodic.** They form 195 contiguous blocks (median ~5 days, max 18), never scattered single points, with no time-of-day pattern.
- **They concentrate in big seas.** Waves >3 m were revised 40% of the time (mean |Δ| 0.63 m) vs ~25% / 0.11 m for 0.5-1.5 m waves, and the largest blocks all fall in the Dec–Mar storm/cyclone season (e.g. 2025-01-27→02-15, 2023-12-01→12-11).
- **Three years are untouched.** 2017, 2018, and 2021 are byte-identical; 2015/16/19/20/22/23/24/25 were republished.

The net effect is that the revised data is rougher:
- 12h autocorrelation dropped 0.85 → 0.74
- persistence RMSE rose from 26.5 cm on the old snapshot to ~40 cm now.

I assume the revision is a data-quality improvement (more accurate storm-period measurements), not a regression — but it means **absolute RMSE is not comparable across snapshots**, while skill-vs-persistence is largely preserved. `test_persistence_baseline_matches_documented_values` pins the current baseline so a future revision is caught rather than silently shifting the headline numbers.


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

## Modeling and Results

> **The one rule.** Decide how you'll judge success, measure the noise in that judgement, and treat any difference smaller than the noise as the noise it is. Because the upstream data is periodically re-derived (see [Upstream revisions](#upstream-revisions)), **absolute RMSE is not portable across data vintages — so every result below leads with _skill versus persistence_**, which is.

### 1. Objective and evaluation contract

Decided *before* modelling:

| | |
|---|---|
| **Target** | `hsig_m` at Mooloolaba, 30-min cadence, indexed at the **forecast origin** `t` |
| **Horizons** | 6 / 12 / 24 / 36 / 48 / 72 h (= 12 / 24 / 48 / 72 / 96 / 144 steps) — a first-class axis |
| **Strategy** | **Direct** (one model per horizon); the sequence NN trains direct per horizon too |
| **Output** | Point forecasts (quantiles/intervals are [roadmap](#future-roadmap) item 1) |
| **Headline metric** | **Skill** `= 1 − RMSE_model / RMSE_persistence` (+ MAE, bias, MASE) |
| **Baselines** | persistence, seasonal-naive, climatology, drift (Theta as a thinned classical check) |
| **Selection** | rolling-origin (5 expanding folds, ~120-day blocks) on **dev = 2015–2024**, mean across folds, fold spread = uncertainty |
| **Embargo** | 144 steps (3 d) dropped at every train/validation seam (purges horizon leakage) |
| **Blind test** | **all of 2025** — pre-committed, refit-and-scored **exactly once** (`confirm.py`) |
| **Noise floor** | moving-block bootstrap CI on every metric; **paired** block bootstrap for every "A beats B" |

Exploratory work (`notebooks/select_backtest.py`) and the confirmatory blind score (`notebooks/confirm.py`) are kept physically separate so the headline can't be tuned.

### 2. Data and provenance

Sources, cleaning, and the ~27%-of-rows republish behaviour are documented under [Data source](#data-source). Because a re-pull silently shifts absolute error, `test_persistence_baseline_matches_documented_values` **pins the current-vintage persistence RMSE** (6/12/24 h) and the climatology crossover — a future revision fails the test instead of quietly moving the headline.

### 3. Exploratory analysis — each figure ends in a decision

| Figure | Reads | Decision |
|---|---|---|
| ![coverage](notebooks/figures/wave_coverage.png) Coverage matrix | Mooloolaba runs 2015→; neighbour buoys switch on in a staircase (Brisbane 2012, Gold Coast/Tweed 2014, Wide Bay 2019) | The all-source common-overlap window only opens in **2019** → source ablation is clipped to it |
| Target distribution (`target_distribution.png`) | right-skewed, heavy upper tail (storm seas), no zero-inflation | **robust (median/IQR) scaling**; no target transform needed for tree/shrinkage models |
| Decomposition (`decomposition.png`) | variance is mostly annual + irregular; ADF p≈0 (stationary in level), strong 1/yr spectral peak, weak 1/day | difference **not** needed; encode **month** seasonality; a climatology baseline is justified |
| ACF/PACF (`acf_pacf.png`) | AR(1)-like decay (ACF≈0.5 at 24 h, ≈0.3 at 48 h), PACF cuts after ~1–2 lags + a small 24 h bump | place lags at the short knees (0.5–3 h) plus daily multiples; expect persistence to fade fast |
| Seasonality calendar (`seasonality_calendar.png`) | annual swing **0.44 m** (Feb storm peak) ≫ diurnal swing **0.05 m** | month features matter, hour-of-day barely; climatology keyed on **month × hour** |
| Lead–lag (`lead_lag.png`) | every neighbour buoy co-moves with the target at **lag 0** (corr 0.70–0.92); wind correlates weakly (≤0.26) with a ~5 h lead | neighbours carry **no forecasting lead** (you don't have their future); only the **primary buoy's own history** and a possible wind edge are candidates |
| Feature × horizon screen (`feature_horizon_screen.png`) | recent `hsig_m`/`hmax_m` + short rolling stats dominate at 6 h; mutual information **decays to ≈0 by 72 h** | short horizons reward recent state; long horizons reward shrinkage toward climatology |

### 4. Baselines and the crossover

Rolling-origin RMSE on dev (skill is vs persistence by definition 0):

| horizon | persistence | seasonal-naive | climatology | drift |
|--------:|----:|----:|----:|----:|
| 6 h  | **0.294** | 0.534 | 0.467 | 0.294 |
| 12 h | **0.403** | 0.534 | 0.467 | 0.403 |
| 24 h | 0.534 | 0.534 | **0.467** | 0.534 |
| 36 h | 0.563 | 0.574 | **0.468** | 0.563 |
| 48 h | 0.574 | 0.574 | **0.468** | 0.574 |
| 72 h | 0.615 | 0.615 | **0.468** | 0.615 |

![baseline crossover](notebooks/figures/baseline_crossover.png)

Persistence wins the short horizons; **climatology (flat ≈0.47 m) overtakes it around 18–24 h** and is the "better baseline" beyond. Locating this crossover tells us what kind of skill is even available: at long range the bar is "beat regression-to-the-seasonal-mean," not "beat the last reading."

### 5. Evaluation protocol & noise floor

Selection is on the **mean of 5 expanding rolling-origin folds** with a 3-day embargo; the **fold spread** (a calm year vs a cyclone year) is often larger than the gap between two models. Each headline RMSE carries a **moving-block bootstrap 95% CI** (blocks longer than the residual autocorrelation scale), and every "A beats B" claim is a **paired block bootstrap** of the per-origin error difference on identical rows — the only way to separate a real gap from the noise when absolute CIs overlap. Tooling: `forecast.backtest.rolling_origin` / `block_bootstrap_ci` / `paired_block_bootstrap`.

### 6. Feature importance & source ablation

The primary buoy's own history carries essentially all the signal. Ridge |coef| share is dominated by `hsig_m` (and its recent rolling stats), then `hmax_m` and `peak_dir`, with `tz_s`/`tp_s`/calendar minor — and `hsig_m`'s share is highest at short horizons, fading as the horizon lengthens.

![importance](notebooks/figures/importance_family.png)

The grouped **source ablation** (Ridge on a uniform reduced grid, **identical 2019–2024 common-overlap rows**) measures whether the neighbour buoys or wind pay their way. RMSE (m), with the marginal change vs the primary-only model:

| horizon | primary | + wind | + neighbours | + all |
|--------:|----:|----:|----:|----:|
| 6 h  | 0.274 | 0.269 (−0.005) | 0.265 (−0.009) | **0.263** (−0.011) |
| 12 h | 0.359 | 0.357 | 0.356 | 0.356 |
| 24 h | **0.440** | 0.445 (+0.005) | 0.451 (+0.011) | 0.458 (+0.018) |
| 72 h | **0.475** | 0.479 (+0.004) | 0.499 (+0.024) | 0.500 (+0.025) |

![source ablation](notebooks/figures/source_ablation.png)

Add-one and drop-one **agree in sign**, so the sources are neither hidden complements nor redundant — they are simply marginal: a **sub-centimetre gain only at 6 h** (well inside the ±2–3 cm fold-spread/bootstrap noise, so not a real finding) and a **real, growing harm from ~24 h onward** as hundreds of mostly-noise columns drag the over-parameterised model below the parsimonious one. **Decision: primary-buoy-only is the feature set.** This is the EDA lead–lag result (neighbours co-move at lag 0; you never hold their future) confirmed by a paired backtest.

### 7. Model selection

The ladder, climbed on dev with `select_backtest.py`: regularised **linear** (Ridge / Lasso / ElasticNet) on the engineered matrix → **gradient-boosted trees** (HistGBM) → **sequence NN** (GRU / LSTM / TCN on raw windows, seed-averaged; GRU won a 1-fold arch screen). Rolling-origin RMSE (m); skill vs persistence in parentheses:

| horizon | persistence | climatology | **Ridge** | **HGB** | GRU | winner |
|--------:|----:|----:|----:|----:|----:|:--|
| 6 h  | 0.294 | 0.467 | 0.271 (+0.08) | **0.257 (+0.13)** | 0.299 (+0.05) | HGB |
| 12 h | 0.403 | 0.467 | 0.355 (+0.12) | **0.349 (+0.13)** | 0.388 (+0.09) | HGB |
| 24 h | 0.534 | 0.467 | 0.440 (+0.18) | **0.432 (+0.19)** | 0.465 (+0.18) | HGB ≈ Ridge |
| 36 h | 0.563 | 0.468 | **0.451 (+0.20)** | 0.465 (+0.17) | 0.484 (+0.19) | Ridge |
| 48 h | 0.574 | 0.468 | **0.456 (+0.21)** | 0.480 (+0.17) | 0.485 (+0.21) | Ridge |
| 72 h | 0.615 | 0.468 | **0.472 (+0.24)** | 0.488 (+0.21) | 0.505 (+0.26) | Ridge |

(Lasso and ElasticNet track Ridge to ±0.01 m — the shrinkage family is interchangeable. GRU is scored on 3 folds, the rest on 5.)

![skill vs horizon](notebooks/figures/skill_vs_horizon.png)

The **family × horizon crossover** is the headline: gradient-boosted trees win the short horizons (rich residual signal to exploit), regularised linear wins the long ones (shrinkage resists the noise the residual has become), with the swap near **24–36 h**. The sequence NN beats persistence everywhere but **loses to both linear and trees at every horizon** — extra capacity buys nothing once the engineered features already expose the autocorrelation structure. At long range every model converges toward the climatology floor (~0.47 m).

### 8. Significance

Read against the noise floor, the *structural* findings survive; the per-horizon point winners mostly do not. Forest plots (per-origin block-bootstrap CIs; colour from the paired test vs persistence) at 6 / 24 / 72 h:

![forest 24h](notebooks/figures/forest_24h.png)

1. **Everything beats persistence at every horizon** — Ridge/Lasso/EN/HGB/GRU/climatology all have paired CIs that exclude zero (green).
2. **Trees short, linear long is real; "which linear/tree" is a tie.** At 24 h the whole ML cluster (HGB 0.432 … Ridge 0.440) sits inside one another's absolute CIs — HGB is nominally best but **tied** with the linear family. The structural swap (HGB ahead at 6–12 h, Ridge ahead at 48–72 h) holds up; the sub-centimetre gaps within a horizon do not.
3. **The sequence NN does not pay for itself** — worse than the linear/tree cluster at every horizon, and far more expensive.
4. **Neighbour buoys and wind do not pay their way** (§6): sub-noise gain at 6 h, real harm beyond.

Because the contenders tie, the **shipping recipe is the simplest member of the tied set: Ridge on primary-buoy features** (one stable model, near-best at every horizon, no seed/early-stopping variance), with HGB an option where the last centimetre at ≤12 h matters. Residual diagnostics show the remaining structure — heteroscedastic errors that widen sharply in big seas and a mean-reversion over-prediction once current `hsig_m` > ~2 m (motivating the [roadmap](#future-roadmap)'s quantile/conformal work):

![residual diagnostics](notebooks/figures/residual_diagnostics.png)

### 9. Blind validation (2025)

The pre-registered candidates were re-fit on dev and scored **exactly once** on the never-touched 2025 slice (16,560 origins). RMSE (m) and skill vs persistence:

| horizon | persistence | climatology | Ridge | HGB | GRU |
|--------:|----:|----:|----:|----:|----:|
| 6 h  | 0.346 | 0.545 | 0.325 (+0.06) | **0.310 (+0.11)** | 0.335 (+0.03) |
| 12 h | 0.490 | 0.545 | 0.436 (+0.11) | **0.416 (+0.15)** | 0.444 (+0.09) |
| 24 h | 0.665 | 0.547 | 0.539 (+0.19) | **0.521 (+0.22)** | 0.529 (+0.20) |
| 36 h | 0.704 | 0.547 | **0.545 (+0.23)** | 0.542 (+0.23) | 0.568 (+0.19) |
| 48 h | 0.716 | 0.548 | **0.548 (+0.24)** | 0.546 (+0.24) | 0.577 (+0.19) |
| 72 h | 0.720 | 0.533 | 0.542 (+0.25) | **0.542 (+0.25)** | 0.576 (+0.20) |

The blind period **reproduces every dev finding**: HGB best at short range, Ridge≈HGB at long range, the GRU last among the learners, and all of them significantly above persistence. Crucially, **2025 was a rougher year** — persistence RMSE is ~0.35→0.72 m (vs 0.29→0.62 on dev) — yet **skill is essentially unchanged** (≈+0.2 from 24 h on, both dev and blind). The absolute error moved with the data vintage; the skill did not. That is the contract working as designed, and it is why the headline is skill, not RMSE.

### 10. Roadmap & negative results

What **didn't** help (first-class results): **neighbour buoys** add nothing to a *forecast* — they co-move with the target contemporaneously (lag 0) and you never hold their future; **wind** offers at most a sub-noise short-horizon edge; richer **gradient-boosted and sequence-NN** models do not beat regularised linear above the noise floor. Forward work is the [Future Roadmap](#future-roadmap): quantile/conformal intervals for the storm-tail underfit the residual diagnostics expose, multi-output `tp_s`/`peak_dir_deg`, and the long-cadence historical bundles.

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
- **`viz`**: Source-agnostic plotting
- **`forecast`**: Target construction, chronological splits, leakage-safe feature engineering, baselines, metrics, rolling-origin backtesting with a block-bootstrap noise floor and paired-significance tests, an evaluation harness, and append-only experiment logging. See *Available forecasters* below for the model list.

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

### Available forecasters

All forecasters share one duck-typed interface — `.fit(X, y) → self`, `.predict(X) → Series`, `.name` — so baselines, linear/tree models, and the sequence NN drop into the same `forecast.backtest.rolling_origin` harness and are scored on identical folds.

| family | classes |
|--------|---------|
| baseline | `Persistence`, `SeasonalNaive`, `SeasonalMean` (climatology), `DriftRandomWalk`, `Theta` |
| linear | `RidgeForecaster`, `LassoForecaster`, `ElasticNetForecaster` (engineered matrix; impute → robust-scale pipeline) |
| trees | `HGBForecaster` (`HistGradientBoostingRegressor`; native NaN handling) |
| sequence NN | `GRUForecaster`, `LSTMForecaster`, `TCNForecaster` via `SeqForecaster` on raw windows; `SeedAverageForecaster` (`forecast.neural`, torch) |
| orchestration | `DirectMultiHorizon` (one model per horizon) |

`build_feature_matrix` produces the leakage-safe engineered matrix (window stats shifted to end at `t−1`, circular-encoded directions, cyclical calendar); `windows.make_windows` produces the raw channel tensors for the NN.

### Logging experiments

`fc.evaluate_and_log(...)` is a drop-in for `fc.evaluate(...)` that appends a record to `experiments.jsonl` (committed at repo root); `fc.log_run(result, ...)` covers results computed outside the harness. Read the log back as a DataFrame with `fc.read_log()`.

### Experiment scripts

All scripts are plain `.py` files — run directly with `./.venv/bin/python notebooks/<script>.py`:

| script | what it does |
|--------|--------------|
| `eda_wave.py`, `eda_wind.py` | coverage matrices + target distribution / decomposition / ACF / seasonality |
| `lead_lag.py`, `seasonality.py`, `feature_screen.py` | cross-source lead–lag, seasonality deep-dive, feature×horizon MI screen |
| `baselines.py` | Phase-3 baselines on dev, the crossover figure, logged to `experiments.jsonl` |
| `select_backtest.py` | **exploratory** ladder (linear → trees → NN) + source ablation, rolling-origin on dev (`--section linear\|trees\|ablation\|nn\|fast\|all`) |
| `confirm.py` | **confirmatory** — refit the frozen recipe, score the 2025 blind slice once |
| `make_figures.py` | rebuild the Phase-11 result figures from the log (`--plot-only` skips model refits) |

## Future Roadmap

1. **Predict quantiles, not just the mean.** Quantile HGB (`HistGradientBoostingRegressor(loss="quantile", quantile=q)`) for P10/P50/P90, or conformalised intervals over Ridge. Also the structural fix for the tail underfit — residuals-vs-predicted shows bias at high wave heights that pinball loss addresses and a target transform won't.

2. **Multi-output forecasts.** 2 m at 90°/14 s breaks differently from 2 m at 150°/8 s. Forecast `tp_s` and `peak_dir_deg` jointly (`MultiOutputRegressor`) so downstream code can apply break-specific transforms.

3. **Long-cadence historical bundles.** Deeper wave history for swell-upstream buoys: Mooloolaba 2000-2014 (1h), Brisbane 1976-2011 (12h), Gold Coast 1987-2014 (6h), Tweed Heads 1995-2011 (1h). Excluded from `qld_ckan.wave.constants.BUOYS` because the pipeline assumes a 30-min axis and these have drifting minute offsets (e.g. 08:55, 14:56). Needs: a cadence parameter on the wave pipeline, snap-to-grid (floor + dedup) before reindex, and a join strategy mixing coarse history with the 30-min grid. Resource IDs at `coastal-data-system-waves-{slug}` on `data.qld.gov.au`. QLD's 1989-1992 DST window forces choosing fixed UTC+10 (`Etc/GMT-10`) or per-row DST.