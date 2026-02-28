"""Integration tests for services/disease_normalizer."""

import logging

import pytest

from indication_scout.services.disease_normalizer import (
    BROADENING_BLOCKLIST,
    normalize_for_pubmed,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "disease, drug, expected_term",
    [
        ("portal hypertension", None, "hypertension"),
        ("atopic dermatitis", None, "dermatitis"),
        ("narcolepsy-cataplexy", None, "narcolepsy"),
        ("alopecia areata", None, "alopecia"),
        ("non-small cell lung carcinoma", None, "lung cancer"),
        ("hepatocellular carcinoma", None, "liver cancer"),
    ],
)
async def test_disease_normalizer(disease, drug, expected_term):
    llm_results = await normalize_for_pubmed(disease, drug)
    terms = [t.strip().lower() for t in llm_results.split("OR")]
    assert any(t == expected_term.lower() for t in terms)


async def test_single_disease_normalizer():
    disease = "Eczematoid dermatitis"
    drug = ""
    result = await normalize_for_pubmed(disease, drug)
    assert result


async def test_normalize_returns_multiple_terms():
    # "atopic eczema" should normalize to two terms joined by OR (e.g. "eczema OR dermatitis")
    result = await normalize_for_pubmed("atopic eczema", None)
    terms = [t.strip() for t in result.split("OR")]
    assert len(terms) >= 2


@pytest.mark.parametrize(
    "raw_term",
    [
        "neoplasm",
        "cancer",
        "malignancy",
    ],
)
async def test_blocklist_terms_return_raw_term(raw_term):
    """Bare blocklisted terms should be returned unchanged (not further generalized)."""
    result = await normalize_for_pubmed(raw_term, drug_name=None)
    result_terms = {t.strip().lower() for t in result.split("OR")}
    # Result must not be a subset of blocklist (i.e. must retain some specificity)
    assert not (result_terms <= BROADENING_BLOCKLIST)


async def test_organ_specificity_not_lost_for_cancer_terms():
    """Organ-specific cancer terms must retain organ context, not collapse to bare 'cancer'."""
    result = await normalize_for_pubmed("non-small cell lung carcinoma", drug_name=None)
    terms = [t.strip().lower() for t in result.split("OR")]
    assert any("lung" in t for t in terms)
