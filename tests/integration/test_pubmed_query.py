"""Integration tests for services/pubmed_query."""

import logging

import pytest

from indication_scout.services.pubmed_query import (
    get_pubmed_query,
    get_disease_synonyms,
)

logger = logging.getLogger(__name__)


async def test_get_pubmed_query_returns_drug_and_term():
    """Result must start with the drug name and contain AND."""
    result = await get_pubmed_query("metformin", "type 2 diabetes mellitus")

    assert result.startswith("metformin AND ")
    assert len(result) > len("metformin AND ")


async def test_get_pubmed_query_edge():
    """Result must start with the drug name and contain AND."""
    result = await get_pubmed_query("bupropion", "narcolepsy-cataplexy syndrome")

    assert result


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
    ],
)
async def test_get_disease_synonyms(disease, synonyms):

    diseases = await get_disease_synonyms(disease)
    is_subset = set(synonyms).issubset(set(diseases))
    assert is_subset


async def test_get_single_disease_synonym():
    disease = "type 2 diabetes nephropathy"
    synonyms = await get_disease_synonyms(disease)

    assert synonyms
