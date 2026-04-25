import logging

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .constants import CKAN_API_BASE, RESOURCE_IDS

logger = logging.getLogger(__name__)

# CKAN Datastore enforces a per-request row cap (32,000 at time of writing);
# using the max minimises round-trips.
_DATASTORE_BATCH = 32000


def _build_session() -> requests.Session:
    """Session with retry/backoff for transient 5xx and connection errors.

    404 is not retried: fetch_all treats it as a legitimate "resource missing"
    signal and skips the year.
    """
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


_SESSION = _build_session()


def _normalize_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Standardise column names to the pre-COLUMN_RENAME_MAP schema.

    Two schema breaks exist across years:
      - All years may carry ' (unit)' suffixes (confirmed from 2022 onwards in
        both CSV files and the Datastore); stripping is a safe no-op otherwise.
      - 2015-2016: wave direction column is 'Dir_Tp TRUE' instead of 'Peak Direction'.
    """
    df = df.rename(columns={col: col.split(" (")[0] for col in df.columns})
    if year < 2017:
        df = df.rename(columns={"Dir_Tp TRUE": "Peak Direction"})
    return df


def fetch_year_datastore(year: int, resource_id: str) -> pd.DataFrame:
    """Fetch all records for one year from the CKAN Datastore API.

    Paginates automatically; each batch is _DATASTORE_BATCH rows.

    Returns a DataFrame with a parsed 'Date/Time' column and standardised
    column names (pre-COLUMN_RENAME_MAP), ready for concatenation.
    """
    records: list[dict] = []
    offset = 0

    while True:
        response = _SESSION.get(
            f"{CKAN_API_BASE}/datastore_search",
            params={"resource_id": resource_id, "limit": _DATASTORE_BATCH, "offset": offset},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()["result"]
        batch = result["records"]
        records.extend(batch)
        if len(records) >= result["total"] or not batch:
            break
        offset += len(batch)

    df = pd.DataFrame(records).drop(columns=["_id"], errors="ignore")
    df = _normalize_columns(df, year)
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="raise")
    return df


def fetch_all(resource_ids: dict[int, str] | None = None) -> list[pd.DataFrame]:
    """Download every year, skipping any that return 404.

    Args:
        resource_ids: year → CKAN resource ID mapping; defaults to RESOURCE_IDS.

    Returns:
        List of per-year DataFrames in chronological order.
    """
    if resource_ids is None:
        resource_ids = RESOURCE_IDS

    frames: list[pd.DataFrame] = []
    for year, rid in resource_ids.items():
        logger.info("Fetching %d...", year)
        try:
            frames.append(fetch_year_datastore(year, rid))
            logger.info("  OK (%d)", year)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("Skipping %d: resource not found", year)
            else:
                raise

    return frames
