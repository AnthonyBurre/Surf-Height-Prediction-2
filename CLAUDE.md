# CLAUDE.md

## Environment
Venv is at `./.venv` (managed by uv). Use `./.venv/bin/python` and `./.venv/bin/pytest` for all Python commands — do not call the system `python3`. To install or change deps, run `uv sync --all-extras` (rebuilds `.venv` from `pyproject.toml` + `uv.lock`); never `pip install` directly.

## Non-obvious points

- **Never Read anything under `data/` directly** — the unified CSV is ~10 MB and will blow up token usage. Inspect data via small Python/pandas scripts.

- **Experiment results go in `experiments.jsonl` at the repo root** — Use `forecast.evaluate_and_log(...)` (drop-in replacement for `evaluate`) or `forecast.log_run(result, ...)` for results computed outside the harness; `forecast.read_log()` returns the file as a DataFrame in one line.