# CLAUDE.md

## Environment

Python venv is at `./.venv`. Use `./.venv/bin/python` and `./.venv/bin/pip` for all Python commands — do not call the system `python3` or `pip`.

## Commands

- `./.venv/bin/python -m qld_ckan wave [--buoy NAME]` — downloads and saves a buoy CSV to `data/{buoy}_wave_data_{years}.csv`. The wind sub-command is `python -m qld_ckan wind [--station NAME]` → `data/{station}_wind_data_{years}.csv`. Package is installed editable (`pip install -e .`), so `src/` is on `sys.path` without a `PYTHONPATH` prefix. If imports fail with `No module named qld_ckan`, re-run `./.venv/bin/pip install -e .`.
- `./.venv/bin/pytest src/tests/ -v` — pytest rootdir discovery + `src/tests/__init__.py` handle path resolution.

## Non-obvious points

- **Index is UTC everywhere downstream of `pipeline.clean`**, even though the raw CKAN records are naive AEST. `datetime_utc` is therefore the index name throughout — no Brisbane-time conversion happens for analysis or plots.

- **Sequence models live in `src/forecast/neural.py`** (`SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`). They window their own input (raw channels + sin/cos direction, no lag/rolling), so pass them the `encode_circular` frame — not the full lag-feature matrix used by the linear/tree models. Requires `torch` from the `forecast` extra (`pip install -e '.[forecast]'`).

- **Never Read anything under `data/` directly** — the unified CSV is ~10 MB and will blow up token usage. Inspect data via small Python/pandas scripts that print summaries (`df.describe()`, `df.head()`, etc.).

- **Rolling features lag-shift by 1 step** in `features.add_rolling_features` — past-only by convention (see its docstring). Enforced by `test_add_rolling_features_are_shifted_by_one`.

- **Skill score is always measured vs. `PersistenceForecaster`.** Persistence is tough to beat at this horizon (12h autocorrelation ≈ 0.8); seasonal-naive and climatology trail it and exist only as diagnostic floors.

- **Experiment results go in `experiments.jsonl` at the repo root** — Use `forecast.evaluate_and_log(...)` (drop-in replacement for `evaluate`) or `forecast.log_run(result, ...)` for results computed outside the harness; `forecast.read_log()` returns the file as a DataFrame in one line.