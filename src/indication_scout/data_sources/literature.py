"""Factory + protocol for the literature-source switch.

Routes ``search`` / ``fetch_abstracts`` / ``get_count`` calls to either
PubMedClient or EuropePMCClient based on ``settings.literature_source``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.europe_pmc import EuropePMCClient
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_pubmed_abstract import PubmedAbstract


@runtime_checkable
class LiteratureClient(Protocol):
    """Common async-context-manager surface for literature clients."""

    cache_dir: Path

    async def __aenter__(self) -> "LiteratureClient": ...
    async def __aexit__(self, *exc: object) -> None: ...

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        date_before: date | None = None,
    ) -> list[str]: ...

    async def get_count(
        self, query: str, date_before: date | None = None
    ) -> int: ...

    async def fetch_abstracts(
        self, pmids: list[str], batch_size: int | None = None
    ) -> list[PubmedAbstract]: ...


def get_literature_client(
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> LiteratureClient:
    """Return the configured literature client (PubMed or Europe PMC)."""
    source = get_settings().literature_source
    if source == "europe_pmc":
        return EuropePMCClient(cache_dir=cache_dir)
    if source == "pubmed":
        return PubMedClient(cache_dir=cache_dir)
    raise ValueError(
        f"Unknown literature_source {source!r}; expected 'pubmed' or 'europe_pmc'"
    )
