# Surf-Height-Prediction-2

Predicts the next-timestep significant wave height (`hsig_m`) for Mooloolaba, Queensland, using wave buoy data from the Queensland Government open data portal (2015–2025).

This is a rework of an earlier project, built around a small, tested Python package (`src/wave_data/`) that replaces ad-hoc notebook downloads.

## Setup

Requires Python 3.14. Create a local virtual environment, install pinned dependencies, and install the package in editable mode:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The `data/` directory is gitignored — generate it by running the pipeline below.

## Running the pipeline

Download every year, clean, and export a unified CSV to `data/mooloolaba_wave_data_2015-2025.csv` (~180k rows, a few minutes over the CKAN Datastore API):

```bash
python -m wave_data
```

Or to a custom location:

```bash
python -m wave_data --output path/to/out.csv
```

## Running tests

```bash
pytest src/tests/ -v
```

Network calls are mocked, so tests run offline.

## Notebooks

```bash
jupyter notebook notebooks/
```

- **`prediction.ipynb`** — Loads the unified CSV, handles missing values, and trains/evaluates Linear Regression, Random Forest, and Gradient Boosting regressors on an 80/20 chronological split.
- **`wave_data_unification.ipynb`** — Legacy. The `src/wave_data/` package supersedes it; prefer the CLI above.

## Dataset schema

The unified CSV has a `datetime_aest` index at 30-minute intervals:

| Column | Description |
|--------|-------------|
| `hsig_m` | Significant wave height (metres) |
| `hmax_m` | Maximum wave height (metres) |
| `tz_s` | Zero-crossing period (seconds) |
| `tp_s` | Peak period (seconds) |
| `peak_dir_deg` | Peak wave direction (degrees) |
| `sst_c` | Sea surface temperature (°C) |

Missing or erroneous readings are encoded as `-99.9` in the raw source files.

## Data source

Queensland Government open data portal, Mooloolaba wave buoy. Fetched via the CKAN Datastore API (`datastore_search`) rather than by downloading raw CSVs, so resource IDs remain stable across portal file renames.

## License

See [LICENSE](LICENSE).
