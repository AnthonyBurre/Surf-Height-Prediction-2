# CLAUDE.md

## Environment
Venv is at `./.venv` (managed by uv). Use `./.venv/bin/python` and `./.venv/bin/pytest` for all Python commands — do not call the system `python3`. To install or change deps, run `uv sync --all-extras` (rebuilds `.venv` from `pyproject.toml` + `uv.lock`); never `pip install` directly.

## Non-obvious points

- **Sequence models live in `src/forecast/neural.py`** (`SimpleRNNForecaster`, `GRUForecaster`, `LSTMForecaster`, `TCNForecaster`). They window their own input (raw channels + sin/cos direction, no lag/rolling), so pass them the `encode_circular` frame — not the full lag-feature matrix used by the linear/tree models. Requires `torch` from the `forecast` extra (installed by default via `uv sync --all-extras`; to install only this extra, `uv sync --extra forecast`).

- **Never Read anything under `data/` directly** — the unified CSV is ~10 MB and will blow up token usage. Inspect data via small Python/pandas scripts.

- **Rolling features lag-shift by 1 step** in `features.add_rolling_features` — past-only by convention (see its docstring). Enforced by `test_add_rolling_features_are_shifted_by_one`.

- **Skill score is always measured vs. `PersistenceForecaster`.**; `ClimatologyHourForecaster` exists as a diagnostic floor and overtakes persistence around h≈24h.

- **Experiment results go in `experiments.jsonl` at the repo root** — Use `forecast.evaluate_and_log(...)` (drop-in replacement for `evaluate`) or `forecast.log_run(result, ...)` for results computed outside the harness; `forecast.read_log()` returns the file as a DataFrame in one line.