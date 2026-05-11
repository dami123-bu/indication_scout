"""
Europe PMC API client.

Three methods, matching the PubMedClient surface so the two are
interchangeable behind a factory:

  1. search         — Find PMIDs matching a query (cached)
  2. fetch_abstracts — Fetch article content for given PMIDs
  3. get_count      — Quick count of results without fetching
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from indication_scout.config import get_settings
from indication_scout.constants import (
    DEFAULT_CACHE_DIR,
    EUROPE_PMC_MAX_CONCURRENT_REQUESTS,
    EUROPE_PMC_PAGE_SIZE_MAX,
    EUROPE_PMC_SEARCH_URL,
    EUROPE_PMC_SOURCE_FILTER,
)
from indication_scout.data_sources.base_client import BaseClient
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)


class EuropePMCClient(BaseClient):
    """Client for querying Europe PMC's REST search API."""

    # Literature is a hard dependency; align with PubMedClient policy so
    # downstream agents don't silently degrade on a transport failure.
    exit_on_retry_exhausted = True

    SEARCH_URL = EUROPE_PMC_SEARCH_URL

    _request_semaphore: asyncio.Semaphore | None = None

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _source_name(self) -> str:
        return "europe_pmc"

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._request_semaphore is None:
            cls._request_semaphore = asyncio.Semaphore(
                EUROPE_PMC_MAX_CONCURRENT_REQUESTS
            )
        return cls._request_semaphore

    @staticmethod
    def _augment_query(query: str, date_before: date | None) -> str:
        """Append the MED source filter and an optional FIRST_PDATE upper bound."""
        parts = [f"({query})", f"AND {EUROPE_PMC_SOURCE_FILTER}"]
        if date_before:
            upper = (date_before - timedelta(days=1)).strftime("%Y-%m-%d")
            parts.append(f"AND FIRST_PDATE:[1900-01-01 TO {upper}]")
        return " ".join(parts)

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        date_before: date | None = None,
    ) -> list[str]:
        """Search Europe PMC and return list of PMIDs."""
        if max_results is None:
            max_results = get_settings().pubmed_search_default_max_results

        cache_params: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "date_before": date_before,
        }
        cached = cache_get("europe_pmc_search", cache_params, self.cache_dir)
        if cached is not None:
            return cached

        effective_query = self._augment_query(query, date_before)
        pmids: list[str] = []
        cursor = "*"
        while len(pmids) < max_results:
            page_size = min(EUROPE_PMC_PAGE_SIZE_MAX, max_results - len(pmids))
            params: dict[str, Any] = {
                "query": effective_query,
                "resultType": "lite",
                "format": "json",
                "pageSize": page_size,
                "cursorMark": cursor,
            }
            async with self._get_semaphore():
                data = await self._rest_get(self.SEARCH_URL, params)

            results = data.get("resultList", {}).get("result", [])
            for r in results:
                pmid = r.get("pmid")
                if pmid:
                    pmids.append(pmid)

            next_cursor = data.get("nextCursorMark")
            if not results or not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        if not pmids:
            logger.warning("Europe PMC search returned 0 PMIDs for query: %r", query)

        cache_set("europe_pmc_search", cache_params, pmids, self.cache_dir)
        return pmids

    async def get_count(self, query: str, date_before: date | None = None) -> int:
        """Quick count of results without fetching full records."""
        params: dict[str, Any] = {
            "query": self._augment_query(query, date_before),
            "resultType": "lite",
            "format": "json",
            # API rejects pageSize=0; use 1 and read hitCount off the envelope.
            "pageSize": 1,
        }
        async with self._get_semaphore():
            data = await self._rest_get(self.SEARCH_URL, params)
        return int(data.get("hitCount", 0))

    async def fetch_abstracts(
        self, pmids: list[str], batch_size: int | None = None
    ) -> list[PubmedAbstract]:
        """Fetch article content for given PMIDs."""
        if not pmids:
            return []

        if batch_size is None:
            batch_size = min(
                get_settings().pubmed_efetch_batch_size, EUROPE_PMC_PAGE_SIZE_MAX
            )

        all_articles: list[PubmedAbstract] = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            # OR of ext_id terms restricted to MED gives unambiguous PMID→record lookup.
            query = " OR ".join(f"ext_id:{p}" for p in batch)
            query = f"({query}) AND {EUROPE_PMC_SOURCE_FILTER}"
            params: dict[str, Any] = {
                "query": query,
                "resultType": "core",
                "format": "json",
                "pageSize": len(batch),
            }
            async with self._get_semaphore():
                data = await self._rest_get(self.SEARCH_URL, params)

            for r in data.get("resultList", {}).get("result", []):
                article = self._parse_result(r)
                if article is not None:
                    all_articles.append(article)

        return all_articles

    @staticmethod
    def _parse_result(r: dict[str, Any]) -> PubmedAbstract | None:
        """Map an Europe PMC `core` result record to a PubmedAbstract."""
        pmid = r.get("pmid")
        if not pmid:
            return None

        authors: list[str] = []
        for a in r.get("authorList", {}).get("author", []):
            name = a.get("fullName") or a.get("lastName")
            if name:
                authors.append(name)

        journal = r.get("journalInfo", {}).get("journal", {}).get("title")

        mesh_terms = [
            m.get("descriptorName")
            for m in r.get("meshHeadingList", {}).get("meshHeading", [])
            if m.get("descriptorName")
        ]

        keywords = [
            k for k in r.get("keywordList", {}).get("keyword", []) if k
        ]

        return PubmedAbstract(
            pmid=pmid,
            title=r.get("title") or "",
            abstract=r.get("abstractText"),
            authors=authors,
            journal=journal,
            pub_date=r.get("firstPublicationDate"),
            mesh_terms=mesh_terms,
            keywords=keywords,
        )
