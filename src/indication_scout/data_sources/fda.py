"""openFDA drug label client and drug name lookup."""

import asyncio
import logging
from pathlib import Path

from indication_scout.config import get_settings
from indication_scout.constants import (
    CACHE_TTL,
    DEFAULT_CACHE_DIR,
    OPENFDA_BASE_URL,
    OPENFDA_LABEL_LIMIT,
)
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()


class FDAClient(BaseClient):
    """Client for querying openFDA drug labels."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        super().__init__()
        self.cache_dir = cache_dir

    @property
    def _source_name(self) -> str:
        return "openfda"

    async def get_label_indications(self, brand_name: str) -> list[str]:
        """Fetch indications_and_usage text from openFDA for a given brand name.

        Queries the drug label endpoint filtered by openfda.brand_name.
        Returns a flat list of indication text strings across all matching labels.
        A 404 response (no results for that brand) returns [] and is cached.

        Args:
            brand_name: Trade/brand name to search (e.g. "Wegovy").

        Returns:
            List of indications_and_usage text strings, possibly empty.
        """
        cache_params = {"brand_name": brand_name.lower()}
        cached = cache_get("fda_label", cache_params, self.cache_dir)
        if cached is not None:
            return cached

        params: dict[str, str | int] = {
            "search": f'openfda.brand_name:"{brand_name}"',
            "limit": OPENFDA_LABEL_LIMIT,
        }
        if _settings.openfda_api_key:
            params["api_key"] = _settings.openfda_api_key

        try:
            data = await self._rest_get(OPENFDA_BASE_URL, params=params)
        except DataSourceError as e:
            if e.status_code == 404:
                cache_set("fda_label", cache_params, [], self.cache_dir, ttl=CACHE_TTL)
                return []
            raise

        results = data.get("results", []) if isinstance(data, dict) else []
        indications: list[str] = []
        for result in results:
            indications.extend(result.get("indications_and_usage", []))

        cache_set("fda_label", cache_params, indications, self.cache_dir, ttl=CACHE_TTL)
        return indications

    async def get_all_label_indications(self, trade_names: list[str]) -> list[str]:
        """Fetch indications from openFDA for all trade names concurrently.

        Fans out get_label_indications for each trade name, deduplicates the
        results, and tolerates individual failures (logs and skips).

        Args:
            trade_names: List of brand/trade names (e.g. ["Ozempic", "Wegovy"]).

        Returns:
            Deduplicated list of indication text strings across all trade names.
        """
        if not trade_names:
            return []

        tasks = [self.get_label_indications(name) for name in trade_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_indications: list[str] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "openFDA label fetch failed for %r: %s", trade_names[i], result
                )
                continue
            all_indications.extend(result)

        return list(dict.fromkeys(all_indications))


async def get_all_drug_names(
    chembl_id: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> list[str]:
    """Return all known names (generic + trade) for a drug by ChEMBL ID.

    Checks the "chembl_drug_names" cache first. On cache miss, calls
    ChEMBLClient.get_all_drug_names to fetch from the API and populate the cache.
    """
    cached = cache_get("chembl_drug_names", {"chembl_id": chembl_id}, cache_dir)
    if cached is not None:
        return cached
    async with ChEMBLClient(cache_dir=cache_dir) as client:
        return await client.get_all_drug_names(chembl_id)
