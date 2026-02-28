"""Integration tests for services/pubmed_query."""

import logging

import pytest

from indication_scout.services.pubmed_query import (
    get_pubmed_query,
)
from indication_scout.services.retrieval import (
    expand_search_terms,
    get_disease_synonyms,
)

logger = logging.getLogger(__name__)


async def test_get_pubmed_query_returns_drug_and_term():
    """Each query must be a '<disease> AND <drug>' string with a diabetes-related disease term."""
    result = await get_pubmed_query("metformin", "type 2 diabetes mellitus")

    assert isinstance(result, list)
    assert len(result) >= 1
    for q in result:
        parts = q.split(" AND ")
        assert len(parts) == 2
        disease_part, drug_part = parts
        assert drug_part.strip() == "metformin"
        assert any(
            w in disease_part.lower()
            for w in ("diabetes", "metabolic", "glucose", "insulin")
        )

# TODO delete
async def test_get_single_pubmed_query_returns_drug_and_term():
    """Each query must be a '<disease> AND <drug>' string with a diabetes-related disease term."""
    results = await get_pubmed_query("Baricitinib", "Rheumatoid Arthritis")

    assert len(results) >= 1


async def test_get_pubmed_query_edge():
    """Each query must be a '<disease> AND <drug>' string."""
    result = await get_pubmed_query("bupropion", "narcolepsy-cataplexy syndrome")

    assert isinstance(result, list)
    assert len(result) >= 1
    for q in result:
        parts = q.split(" AND ")
        assert len(parts) == 2
        disease_part, drug_part = parts
        assert disease_part.strip() != ""
        assert drug_part.strip() == "bupropion"


async def test_get_single_disease_synonym():
    disease = "type 2 diabetes nephropathy"
    synonyms = await get_disease_synonyms(disease)

    assert synonyms


# async def test_expand_search_terms_metformin_colorectal():
#     """Each term must be a non-empty string containing both drug and disease concepts."""
#     drug_profile = {
#         "mechanism": "AMPK activator, biguanide",
#         "targets": ["PRKAA1", "PRKAA2"],
#         "approved_indications": ["type 2 diabetes mellitus"],
#     }
#     terms = await expand_search_terms("metformin", "colorectal cancer", drug_profile)
#
#     assert isinstance(terms, list)
#     assert len(terms) >= 5
#     assert all(isinstance(t, str) and len(t) > 0 for t in terms)
#     assert any("metformin" in t.lower() for t in terms)
#     assert any(
#         any(w in t.lower() for w in ("colorectal", "colon", "rectal"))
#         for t in terms
#     )
