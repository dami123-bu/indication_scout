"""Integration tests for services/pubmed_query."""

import logging

from indication_scout.services.pubmed_query import get_pubmed_query

logger = logging.getLogger(__name__)


async def test_get_pubmed_query_returns_drug_and_term():
    """Result must start with the drug name and contain AND."""
    result = await get_pubmed_query("metformin", "type 2 diabetes mellitus")

    assert result.startswith("metformin AND ")
    assert len(result) > len("metformin AND ")
