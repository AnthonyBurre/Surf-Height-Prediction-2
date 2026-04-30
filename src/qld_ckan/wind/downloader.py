import pandas as pd

import qld_ckan

from .constants import RESOURCE_IDS

# Re-exported so ``patch("qld_ckan.wind.downloader._session")``-style tests can
# replace the session for this sub-package without affecting the wave side.
_session = qld_ckan.session
_DATASTORE_BATCH = qld_ckan.DATASTORE_BATCH


def fetch_year_datastore(year: int, resource_id: str) -> pd.DataFrame:
    """Fetch all records for one year from the CKAN Datastore API.

    Returns a DataFrame with raw column names as exposed by the datastore;
    cleaning (rename, timestamp build, regrid) happens in pipeline.clean.
    """
    records = qld_ckan.paginate_records(_session(), resource_id)
    return pd.DataFrame(records).drop(columns=["_id"], errors="ignore")


def fetch_all(resource_ids: dict[int, str] | None = None) -> list[pd.DataFrame]:
    """Download every year, skipping any that return 404."""
    if resource_ids is None:
        resource_ids = RESOURCE_IDS
    return qld_ckan.fetch_all_years(resource_ids, fetch_year_datastore)
