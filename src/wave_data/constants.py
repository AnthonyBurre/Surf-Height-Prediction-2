CKAN_API_BASE = "https://www.data.qld.gov.au/api/3/action"

# All years are served via the CKAN Datastore API. Resource IDs are stable
# even when the portal renames or replaces the underlying file.
RESOURCE_IDS: dict[int, str] = {
    2015: "81df149b-67fc-4e5c-8ab8-b479001e04eb",
    2016: "8e7fcf48-ac17-45ea-b4f3-b30cd1739658",
    2017: "61faa02e-a4e5-400e-98ff-3266b255da1b",
    2018: "b92629f7-a79f-45a2-857d-2217b2f11e63",
    2019: "d80243cd-5bca-41a0-b0f8-4d4fac168d94",
    2020: "f58445e2-44cc-48e6-ad99-1ee56e1ee402",
    2021: "b700d6d6-31fa-43b1-ae89-1fa811765aff",
    2022: "c995189b-82ec-422b-bf56-ccc5510618bc",
    2023: "fbad2855-d2cb-4889-ba1e-bee2c042d1c2",
    2024: "51c8f862-5ecd-4cff-af35-968ddd48a16e",
    2025: "c0cfd5c8-59c1-4fcf-b9d9-75bce4149808",
}

# Raw source files use -99.9 to indicate missing/erroneous readings
SENTINEL_VALUE = -99.9

# Final standardised column names applied after per-year normalisation
COLUMN_RENAME_MAP: dict[str, str] = {
    "Date/Time": "datetime_aest",
    "Hs": "hsig_m",
    "Hmax": "hmax_m",
    "Tz": "tz_s",
    "Tp": "tp_s",
    "Peak Direction": "peak_dir_deg",
    "SST": "sst_c",
}
