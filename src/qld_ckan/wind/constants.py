# Hourly air-quality + meteorology stations from the QLD Government CKAN portal.
# Resource IDs are stable across portal file renames. Mountain Creek is the
# default: it sits at -26.6917, 153.1038 — effectively co-located with the
# Mooloolaba wave buoy — and carries a 10 m ultrasonic wind sensor referenced
# to true north. Each yearly resource is a separate package under the
# `air-quality-monitoring-{year}` slug.
STATIONS: dict[str, dict[int, str]] = {
    "mountain-creek": {
        2015: "9e04a2ef-855d-49e9-b252-a1f46dc576ac",
        2016: "aa9b6fc8-0cd3-4f05-9594-4678d2ba2828",
        2017: "5fafabf3-76d6-4b72-b5c4-cef2aa7f18a6",
        2018: "858b8717-37e2-47ca-b7bf-f6c16372db83",
        2019: "4b982a1f-d9be-4e4a-8ada-6e17aba23fda",
        2020: "522f0358-7435-418b-ad87-4365c2c57da4",
        2021: "d36f01f5-b7de-49c9-a784-7d54afc69f72",
        2022: "aa1441e2-3ef6-4690-acf3-6111932810e1",
        2023: "474d80f7-f859-4d9f-8e20-f39c3fdf0800",
        2024: "f0199e4f-a10a-4f7a-9fb0-f1eedef674ad",
    },
    "deception-bay": {
        2015: "be0ba961-456e-451d-842d-d62fe7a85ae8",
        2016: "16fc598d-6dae-4c5e-a69b-580f698530aa",
        2017: "96de5878-0df8-4b8b-9153-70eb52152af9",
        2018: "98f5b5bc-0c65-45ab-83a3-585d330d5632",
        2019: "68cbb103-301b-47d5-ae2b-d81f56fc7c9c",
        2020: "790c2a25-9400-431a-bc66-fd24ce30adab",
        2021: "c003b168-40f5-4595-b580-4f4d1aa4c37f",
        2022: "29603390-0a47-40dd-b4b8-97137cabcd3c",
        2023: "a2d5dd4b-6e2a-410e-919c-3e589c8367bf",
        2024: "533fb7e9-2238-4e3b-82d0-96162d975366",
    },
}

# Backwards-compatible alias for the primary Mountain Creek station.
RESOURCE_IDS: dict[int, str] = STATIONS["mountain-creek"]

# Wind columns kept after cleaning. Pollutant fields (ozone, NOx, PM10, etc.)
# and the air-temperature column (which is absent from later years) are dropped
# at clean time so the unified frame has a stable, wind-focused schema.
COLUMN_RENAME_MAP: dict[str, str] = {
    "Wind Direction (degTN)": "wind_dir_deg",
    "Wind Speed (m/s)": "wind_speed_ms",
    "Wind Sigma Theta (deg)": "wind_sigma_theta_deg",
    "Wind Speed Std Dev (m/s)": "wind_speed_std_ms",
}
