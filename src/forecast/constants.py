"""Project-wide constants for the forecasting harness.

Single source of truth so notebooks and tests never hard-code horizons, column
names, or paths. Everything is indexed at the *forecast origin* `t` on the
30-minute Brisbane grid; a horizon in hours maps to a number of 30-minute steps
via :data:`HORIZON_STEPS`.
"""
from pathlib import Path

# Repo root = .../Surf-Height-Prediction-2 (this file is src/forecast/constants.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
LOG_PATH = REPO_ROOT / "experiments.jsonl"
FIGURE_DIR = REPO_ROOT / "notebooks" / "figures"

# --- Target & cadence -------------------------------------------------------
TARGET_BUOY = "mooloolaba"
TARGET_COL = "hsig_m"
CADENCE = "30min"
STEPS_PER_HOUR = 2
SOURCE_TZ = "Australia/Brisbane"  # fixed UTC+10, no DST

# --- Horizons (the first-class axis) ----------------------------------------
HORIZONS_H = (6, 12, 24, 36, 48, 72)
# Hours -> 30-minute steps ahead.
HORIZON_STEPS = {h: h * STEPS_PER_HOUR for h in HORIZONS_H}

# --- Columns ----------------------------------------------------------------
WAVE_COLS = ["hsig_m", "hmax_m", "tz_s", "tp_s", "peak_dir_deg", "sst_c"]
WIND_COLS = ["wind_dir_deg", "wind_speed_ms", "wind_sigma_theta_deg", "wind_speed_std_ms"]
# Angular columns -> circular (sin, cos) encoding in feature engineering.
WAVE_DIR_COLS = ["peak_dir_deg"]
WIND_DIR_COLS = ["wind_dir_deg"]

# --- Seasonality ------------------------------------------------------------
STEPS_PER_DAY = 48          # 30-min steps in a day (diurnal seasonal-naive period)
STEPS_PER_YEAR = 48 * 365   # approximate annual period for climatology

# --- Evaluation contract ----------------------------------------------------
# Pre-committed blind slice: all of 2025 is reserved and scored exactly once.
BLIND_START = "2025-01-01"

# Default exogenous pairings (buoy/station that sits closest to the target).
WIND_PAIR = "mountain-creek"  # pairs with the Mooloolaba buoy
NEIGHBOUR_BUOYS = [
    "caloundra", "brisbane", "north-moreton-bay",
    "gold-coast", "tweed-heads", "palm-beach", "wide-bay",
]
