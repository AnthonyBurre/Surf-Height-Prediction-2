import logging
from functools import cache

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .constants import CKAN_API_BASE, RESOURCE_IDS

logger = logging.getLogger(__name__)

# CKAN Datastore enforces a per-request row cap (32,000 at time of writing);
# using the max minimises round-trips.
_DATASTORE_BATCH = 32000


@cache
def _session() -> requests.Session:
    """Singleton session with retry/backoff for transient 5xx and connection errors.

    Built on first use — no HTTP machinery is constructed at import time.
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


def fetch_year_datastore(year: int, resource_id: str) -> pd.DataFrame:
    """Fetch all records for one year from the CKAN Datastore API.

    Paginates automatically; each batch is _DATASTORE_BATCH rows.

    Returns a DataFrame with raw column names as exposed by the datastore;
    cleaning (rename, timestamp build, regrid) happens in pipeline.clean.
    """
    records: list[dict] = []
    offset = 0

    while True:
        response = _session().get(
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
