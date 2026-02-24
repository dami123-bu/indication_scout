"""Integration tests for services/retrieval."""

import logging

from indication_scout.services.retrieval import fetch_and_cache, semantic_search

logger = logging.getLogger(__name__)


# async def test_fetch_and_cache_returns_pmids():
#     """Should return a non-empty list of PMID strings for valid queries."""
#     queries = [
#         "metformin AND colorectal cancer",
#         "metformin AND AMPK AND colon",
#     ]
#     pmids = await fetch_and_cache(queries)
#
#     assert isinstance(pmids, list)
#     assert len(pmids) >= 1
#     assert all(isinstance(p, str) and p.isdigit() for p in pmids)


# async def test_fetch_and_cache_deduplicates():
#     """Duplicate queries should not produce duplicate PMIDs."""
#     queries = [
#         "metformin AND colorectal cancer",
#         "metformin AND colorectal cancer",
#     ]
#     pmids = await fetch_and_cache(queries)
#
#     assert len(pmids) == len(set(pmids))


# async def test_semantic_search_returns_ranked_abstracts():
#     """Should return up to top_k dicts with required abstract fields."""
#     results = await semantic_search("metformin inhibits mTOR in colon cancer", top_k=5)
#
#     assert isinstance(results, list)
#     assert len(results) <= 5
#     assert len(results) >= 1
#     first = results[0]
#     assert isinstance(first["pmid"], str) and first["pmid"].isdigit()
#     assert isinstance(first["title"], str) and len(first["title"]) > 0
#     assert isinstance(first["abstract"], str) and len(first["abstract"]) > 0
#     assert isinstance(first["score"], float)


# async def test_semantic_search_respects_top_k():
#     """Result count must not exceed top_k."""
#     results = await semantic_search("metformin colorectal cancer chemoprevention", top_k=3)
#
#     assert len(results) <= 3
