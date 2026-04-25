# CLAUDE.md

## Environment

Python venv is at `./.venv`. Use `./.venv/bin/python` and `./.venv/bin/pip` for all Python commands — do not call the system `python` or `pip`.

## Commands

- `./.venv/bin/python -m wave_data` — package is installed editable (`pip install -e .`), so `src/` is on `sys.path` without a `PYTHONPATH` prefix. If imports fail with `No module named wave_data`, re-run `./.venv/bin/pip install -e .`.
- `./.venv/bin/pytest src/tests/ -v` — pytest rootdir discovery + `src/tests/__init__.py` handle path resolution.

## Non-obvious points

- **Index is UTC everywhere downstream of `pipeline.clean`**, even though the raw CKAN records are naive AEST. `clean()` localises to `Australia/Brisbane` (a fixed UTC+10 since Queensland doesn't observe DST) and immediately `tz_convert("UTC")` so the CSV, modelling pipeline, and viz all share one tz. `datetime_utc` is therefore the index name throughout — no Brisbane-time conversion happens for analysis or plots. `add_time_features` and `ClimatologyHourForecaster` read `index.hour` in UTC; the sin/cos cyclical encoding is phase-invariant so a 10-hour rotation has no effect on model skill.

- **Column normalisation runs per-year, before concat** (`downloader._normalize_columns`). Two schema breaks drive this: pre-2017 uses `Dir_Tp TRUE` instead of `Peak Direction`; 2022+ carries ` (unit)` suffixes on every column.

- **`fetch_all` skips 404s silently, re-raises every other `HTTPError`.** A missing year doesn't kill the run, but a 500 will.

- **Data comes from the CKAN Datastore API, not CSV downloads.** `RESOURCE_IDS` are stable across portal file renames, which is why this replaced the older `DATA_URLS` dict preserved in the legacy notebook.

- **Multi-buoy fetches go through `BUOYS` in `wave_data/constants.py`** — a `dict[slug, dict[year, resource_id]]`. `RESOURCE_IDS` is an alias of `BUOYS["mooloolaba"]` (the primary forecasting target). Neighbouring buoys (Caloundra, Brisbane, Gold Coast) carry **2024-2025 only** — enough for cross-source correlation analysis without a half-hour cold download. Fetch a neighbour with `pipeline.run(output_path=..., resource_ids=BUOYS[slug])`; the `python -m wave_data` CLI still hardcodes Mooloolaba.

- **Chronological 80/20 split, not random** — deliberate, to avoid leakage in the time series. `forecast.chronological_split` is the reusable helper.

- **`src/forecast/` is the modelling package**, separate from `src/wave_data/` (ETL). Flat import surface: `from forecast import load_data, make_target, chronological_split, PersistenceForecaster, evaluate, compare, ...`. Experiments live in `notebooks/forecast_comparison.ipynb`.

- **Sequence models live in `src/forecast/neural.py`** (`SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`). They window their own input (raw channels + sin/cos direction, no lag/rolling), so pass them the `encode_circular` frame — not the full lag-feature matrix used by the linear/tree models. Requires `torch` from the `notebooks` extra.

- **Plotting lives in `src/viz/`**, separate from `forecast/` on purpose: it's source-agnostic so future data sources (other buoys, BOM/GFS atmospheric grids) reuse it without pulling in modelling deps. Multi-source functions take `dict[str, pd.Series | pd.DataFrame]` keyed by source label; that label flows into legends and heatmap titles. Notebook-level `plt.plot(...)` calls should migrate here if they're worth keeping.

- **Never Read `.ipynb` files or anything under `data/` directly** — executed notebooks carry huge inline base64 chart outputs, and the unified CSV is ~10 MB. Both blow up token usage. The `Read` tool is blocked on those paths by `.claude/settings.json`. Inspect notebooks via `jupyter nbconvert --to script --stdout <path>` (strips outputs); inspect data via small Python/pandas scripts that print summaries (`df.describe()`, `df.head()`, etc.). Wholesale notebook replacement: `rm` then Write, or Python JSON dump via Bash.

- **Forecast target is indexed at the origin `t`, not the prediction time `t+h`.** `make_target()` returns `hsig_m.shift(-HORIZON_STEPS)` so `y.loc[t]` is the value *at* `t + 12h`. The last `HORIZON_STEPS` rows of `y` are NaN — `evaluate()` masks them, direct `model.fit` callers must mask.

- **Rolling features lag-shift by 1 step** in `features.add_rolling_features`. Without this the window includes the current observation, leaking the label. Enforced by `test_add_rolling_features_are_shifted_by_one`.

- **Skill score is always measured vs. `PersistenceForecaster`.** Persistence is tough to beat at this horizon (12h autocorrelation ≈ 0.8); seasonal-naive and climatology trail it and exist only as diagnostic floors.
