# CLAUDE.md

## Environment

Python venv is at `./.venv`. Use `./.venv/bin/python` and `./.venv/bin/pip` for all Python commands — do not call the system `python` or `pip`.

## Commands

- `./.venv/bin/python -m wave_data` — package is installed editable (`pip install -e .`), so `src/` is on `sys.path` without a `PYTHONPATH` prefix. If imports fail with `No module named wave_data`, re-run `./.venv/bin/pip install -e .`.
- `./.venv/bin/pytest src/tests/ -v` — pytest rootdir discovery + `src/tests/__init__.py` handle path resolution.

## Non-obvious points

- **Index is UTC everywhere downstream of `pipeline.clean`**, even though the raw CKAN records are naive AEST. `datetime_utc` is therefore the index name throughout — no Brisbane-time conversion happens for analysis or plots.

- **Sequence models live in `src/forecast/neural.py`** (`SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`). They window their own input (raw channels + sin/cos direction, no lag/rolling), so pass them the `encode_circular` frame — not the full lag-feature matrix used by the linear/tree models. Requires `torch` from the `notebooks` extra.

- **Never Read `.ipynb` files or anything under `data/` directly** — executed notebooks carry huge inline base64 chart outputs, and the unified CSV is ~10 MB. Both blow up token usage. The `Read` tool is blocked on those paths by `.claude/settings.json`. Inspect notebooks via `jupyter nbconvert --to script --stdout <path>` (strips outputs); inspect data via small Python/pandas scripts that print summaries (`df.describe()`, `df.head()`, etc.). Wholesale notebook replacement: `rm` then Write, or Python JSON dump via Bash.

- **Rolling features lag-shift by 1 step** in `features.add_rolling_features` — past-only by convention (see its docstring). Enforced by `test_add_rolling_features_are_shifted_by_one`.

- **Skill score is always measured vs. `PersistenceForecaster`.** Persistence is tough to beat at this horizon (12h autocorrelation ≈ 0.8); seasonal-naive and climatology trail it and exist only as diagnostic floors.

- **Experiment results go in `experiments.jsonl` at the repo root** — one JSON record per run with `{timestamp, git_sha, name, model_class, hyperparams, data_sources, n_features, train, test, metrics, extra}`. Use `forecast.evaluate_and_log(...)` (drop-in replacement for `evaluate`) or `forecast.log_run(result, ...)` for results computed outside the harness; `forecast.read_log()` returns the file as a DataFrame in one line. The file is committed; `git_sha` (with a `-dirty` suffix when the working tree is dirty) makes any row reproducible by checkout.