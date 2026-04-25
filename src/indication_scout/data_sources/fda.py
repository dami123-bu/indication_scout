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

    async def get_label_indications(self, drug_name: str) -> list[str]:
        """Fetch indications_and_usage text from openFDA for a given drug name.

        Queries the drug label endpoint, matching either openfda.brand_name OR
        openfda.generic_name. This catches labels where the name is registered
        only under generic_name (e.g. ~5% of metformin labels).
        Returns a flat list of indication text strings across all matching labels.
        A 404 response (no results) returns [] and is NOT cached, so a later
        run can re-query in case the absence was transient.

        Args:
            drug_name: Any drug name — trade name, generic/INN, USAN, etc.
                       (e.g. "Wegovy", "metformin").

        Returns:
            List of indications_and_usage text strings, possibly empty.
        """
        cache_params = {"drug_name": drug_name.lower()}
        cached = cache_get("fda_label", cache_params, self.cache_dir)
        if cached is not None:
            return cached

        params: dict[str, str | int] = {
            "search": (
                f'(openfda.brand_name:"{drug_name}"'
                f' OR openfda.generic_name:"{drug_name}")'
            ),
            "limit": OPENFDA_LABEL_LIMIT,
        }
        if _settings.openfda_api_key:
            params["api_key"] = _settings.openfda_api_key

        try:
            data = await self._rest_get(OPENFDA_BASE_URL, params=params)
        except DataSourceError as e:
            if e.status_code == 404:
                return []
            raise

        results = data.get("results", []) if isinstance(data, dict) else []
        indications: list[str] = []
        for result in results:
            indications.extend(result.get("indications_and_usage", []))

        cache_set("fda_label", cache_params, indications, self.cache_dir, ttl=CACHE_TTL)
        return indications

    async def get_all_label_indications(self, drug_names: list[str]) -> list[str]:
        """Fetch indications from openFDA for all drug names concurrently.

        Fans out get_label_indications for each name (trade or generic/INN),
        deduplicates results, and tolerates individual failures (logs and skips).

        Args:
            drug_names: List of any drug names — trade, generic/INN, USAN, etc.
                        (e.g. ["Ozempic", "Wegovy", "semaglutide"]).

        Returns:
            Deduplicated list of indication text strings across all names.
        """
        if not drug_names:
            return []

        tasks = [self.get_label_indications(name) for name in drug_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_indications: list[str] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "openFDA label fetch failed for %r: %s", drug_names[i], result
                )
                continue
            all_indications.extend(result)

        return list(dict.fromkeys(all_indications))
