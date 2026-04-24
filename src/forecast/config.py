"""Shared constants for the forecast package.

Everything downstream keys off ``HORIZON_STEPS`` so changing the lead time
(e.g. 6h, 24h) only requires flipping one number.
"""

# Wave buoy observations arrive every 30 minutes. A 12-hour forecast horizon
# is therefore 24 steps ahead.
SAMPLING_FREQ_MINUTES = 30
HORIZON_HOURS = 12
HORIZON_STEPS = HORIZON_HOURS * 60 // SAMPLING_FREQ_MINUTES  # 24

TARGET_COL = "hsig_m"

FEATURE_COLS = [
    "hsig_m",
    "hmax_m",
    "tz_s",
    "tp_s",
    "peak_dir_deg",
    "sst_c",
]

# Wave direction is a compass bearing — jumps from 359° to 1° are 2° apart,
# not 358°. Any model that treats it as a plain float will learn that
# discontinuity as signal. Encode as (sin, cos) instead.
CIRCULAR_COLS = ["peak_dir_deg"]
