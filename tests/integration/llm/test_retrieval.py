"""Integration tests for services/retrieval."""

import logging

import pytest

from indication_scout.services.retrieval import (
    get_disease_synonyms,
    fetch_and_cache,
    semantic_search,
    synthesize,
    expand_search_terms,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "disease, synonyms",
    [
        (
            "eczematoid dermatitis",
            ["eczematoid dermatitis", "eczema", "AD", "atopic dermatitis"],
        ),
        (
            "benign prostatic hyperplasia",
            [
                "benign prostatic hyperplasia",
                "BPH",
                "enlarged prostate",
                "prostatic enlargement",
            ],
        ),
        (
            "HER2-positive breast cancer",
            ["breast cancer", "ERBB2-positive breast cancer"],
        ),
        (
            "CML",
            ["CML", "chronic myeloid leukemia"],
        ),
    ],
)
async def test_get_disease_synonyms(disease, synonyms):
    """Returned synonyms should include all expected terms for the given disease."""
    result = await get_disease_synonyms(disease)

    assert set(synonyms).issubset(set(result))


@pytest.mark.parametrize(
    "disease, expected_terms",
    [
        ("type 2 diabetes", ["diabetes"]),
        ("chronic myeloid leukemia", ["leukemia", "CML"]),
        ("HER2-positive breast cancer", ["breast cancer", "HER2"]),
        ("benign prostatic hyperplasia", ["BPH", "prostat"]),
    ],
)
async def test_get_best_disease_contains_key_term(disease, expected_terms):
    """Returned PubMed term should contain at least one expected keyword."""
    result = await get_best_disease(disease)

    assert any(term.lower() in result.lower() for term in expected_terms)


async def test_get_best_disease():
    disease = "renal tubular dysgenesis"

    """Returned PubMed term should contain at least one expected keyword."""
    result = await get_best_disease(disease)

    assert result


# --- Stubs for unimplemented methods ---


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
#     results = await semantic_search("colorectal cancer", "metformin", top_k=5)
#
#     assert isinstance(results, list)
#     assert 1 <= len(results) <= 5
#     first = results[0]
#     assert isinstance(first["pmid"], str) and first["pmid"].isdigit()
#     assert isinstance(first["title"], str) and len(first["title"]) > 0
#     assert isinstance(first["abstract"], str) and len(first["abstract"]) > 0
#     assert isinstance(first["similarity"], float)
#     assert 0.0 <= first["similarity"] <= 1.0


# async def test_semantic_search_respects_top_k():
#     """Result count must not exceed top_k."""
#     results = await semantic_search("colorectal cancer", "metformin", top_k=3)
#
#     assert len(results) <= 3


# async def test_synthesize_returns_structured_summary():
#     """Should return a structured evidence summary with PMIDs."""
#     abstracts = [
#         {
#             "pmid": "29734553",
#             "title": "Metformin suppresses colorectal cancer",
#             "abstract": "Metformin activates AMPK and inhibits mTOR in colon cancer cells.",
#             "similarity": 0.89,
#         }
#     ]
#     result = synthesize("metformin", "colorectal cancer", abstracts)
#
#     assert result is not None
#     assert "pmid" in str(result) or hasattr(result, "pmids")


# async def test_expand_search_terms_returns_queries():
#     """Should return a list of PubMed-style query strings."""
#     drug_profile = {"name": "metformin", "mechanism": "AMPK activator"}
#     queries = await expand_search_terms("metformin", "colorectal cancer", drug_profile)
#
#     assert isinstance(queries, list)
#     assert len(queries) >= 1
#     assert all(isinstance(q, str) and len(q) > 0 for q in queries)
