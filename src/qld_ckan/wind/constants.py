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
    "southport": {
        # Gold Coast coastal station; pairs with the gold-coast and palm-beach
        # wave buoys. Deployed mid-2018, so 2015-2017 are unavailable.
        2018: "3fb172bb-8ce6-4246-9c57-6bbffcaebb58",
        2019: "6006b302-b718-49d2-ba20-ddebac030208",
        2020: "2046d2c0-f7ab-460c-950a-d9285d5aab40",
        2021: "fcc1cf65-1265-4934-9625-1162b89760b1",
        2022: "2986eca8-8d12-4f6e-8c7e-e4f4796aaccf",
        2023: "a2866895-f7cb-4f2a-be0f-b5a82eea3672",
        2024: "f46541c1-4083-46f4-aea5-14ad745580b4",
    },
    "lytton": {
        # At the mouth of the Brisbane River, due east of Brisbane CBD; pairs
        # with the brisbane wave buoy. Full 2015-2024 coverage.
        2015: "1ad0c355-375b-4a0b-8605-504b2dfb067f",
        2016: "7d26b5d7-6d54-4e52-ae79-0f1590807e06",
        2017: "8f6bc380-1a5f-4fc2-82bf-3615136b532a",
        2018: "51c9a6f2-2d55-4843-a173-96d6d8ecfc15",
        2019: "54b03a65-95b2-4f17-8027-67196d04c2df",
        2020: "fff02ccb-3507-4764-b7da-95ac97d2cb27",
        2021: "a8331064-14e7-4d28-b513-e50106424c85",
        2022: "994cd1f1-72ae-42c2-b5df-46660d3c9ce6",
        2023: "7bbda45f-6dfe-440f-8af9-718feea7e8fb",
        2024: "0b90eccd-bce5-4da6-8c7a-49fb8267cd06",
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
