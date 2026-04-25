CKAN_API_BASE = "https://www.data.qld.gov.au/api/3/action"

# All years are served via the CKAN Datastore API. Resource IDs are stable
# even when the portal renames or replaces the underlying file.
#
# BUOYS is keyed by short slug; each value maps year → CKAN resource ID.
# Mooloolaba is the primary forecasting target; the southern neighbours
# (Caloundra, Brisbane, Gold Coast) span 2024-2025 only — enough for
# cross-source correlation analysis without a half-hour cold download.
BUOYS: dict[str, dict[int, str]] = {
    "mooloolaba": {
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
    },
    "caloundra": {
        # 2013 resource is a bundle covering 2013-2015; pre-2017 column schema.
        2013: "fe51b69b-df4e-4eb2-981f-3e18707bd09f",
        2016: "be62134c-15a2-4736-8b2a-bbfd323f10ce",
        2017: "80481a30-ac2d-4e67-82e9-d5fcb282f58b",
        2018: "e302c519-77ec-46e5-b819-767ac177f37a",
        2019: "6603ad4e-98cd-4215-877c-61b9759cbbd0",
        2020: "fdd1bc53-1d59-46bd-a5a6-90c18f851bd8",
        2021: "43354fd8-4a9b-4004-b42f-2fcb9e8d5b8a",
        2022: "a82ce37b-cca5-44db-a2f1-14c2b6cea836",
        2023: "f7a3f62e-4021-4d39-9fdd-4e13b5d62223",
        2024: "a7d465ba-0cab-4b4e-a93c-4c8c86c196b5",
        2025: "d276f08f-5853-493e-a607-0dfa60f6e850",
    },
    "brisbane": {
        2024: "7f7da919-6c68-4bf4-bb4b-b6a0d936316c",
        2025: "60abceb3-2949-48df-8181-3f98ae72108d",
    },
    "gold-coast": {
        2015: "30cdfd68-52e9-4c5c-933c-03c2fed5a11a",
        2016: "c5e598d5-5a9a-45be-9da6-a9f47042e006",
        2017: "f85931b4-926a-49e3-9e56-d65bd49a9f14",
        2018: "d1049f97-45a9-4b3d-80be-7fce8cd3ec29",
        2019: "ee5859f9-e55c-434f-b20c-7da30b1e53e1",
        2020: "73b4e42a-f3e7-4632-8b8e-d52205899048",
        2021: "edc414a8-3ffa-47a2-9d9b-14f07eb22072",
        # 2022 Mk4 (e0068000-…) skipped — empty Date/Time fields in datastore.
        2022: "2eeb6b4d-f52a-45c1-b640-336aaf53b40b",
        2023: "618d4d1e-fa39-4e04-929a-94ca6e107973",
        2024: "67c8cf49-cb29-4cd1-86ff-d61bfdb8cbba",
        2025: "a8a12129-c99d-45f6-832b-a5cee4754b54",
    },
}

# Backwards-compatible alias for the primary Mooloolaba buoy.
RESOURCE_IDS: dict[int, str] = BUOYS["mooloolaba"]

# Raw source files use -99.9 to indicate missing/erroneous readings
SENTINEL_VALUE = -99.9

# Final standardised column names applied after per-year normalisation
COLUMN_RENAME_MAP: dict[str, str] = {
    "Date/Time": "datetime_utc",
    "Hs": "hsig_m",
    "Hmax": "hmax_m",
    "Tz": "tz_s",
    "Tp": "tp_s",
    "Peak Direction": "peak_dir_deg",
    "SST": "sst_c",
}
