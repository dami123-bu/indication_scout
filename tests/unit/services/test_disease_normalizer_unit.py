"""Unit tests for disease_normalizer blocklist logic."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.services.disease_normalizer import (
    BROADENING_BLOCKLIST,
    normalize_for_pubmed,
)

logger = logging.getLogger(__name__)


# ── Blocklist guard on initial normalization ──────────────────────────────────


@pytest.mark.parametrize(
    "raw_term, llm_output",
    [
        ("neoplasm", "cancer"),
        ("malignant neoplasm", "neoplasm"),
        ("tumor", "tumor"),
        ("malignancy", "malignancy"),
        ("cancer OR neoplasm", "cancer OR neoplasm"),
    ],
)
async def test_blocklist_rejects_overly_broad_initial_normalization(
    raw_term, llm_output
):
    """If LLM returns only blocklisted terms, raw_term is returned unchanged."""
    with patch(
        "indication_scout.services.disease_normalizer.llm_normalize",
        new=AsyncMock(return_value=llm_output),
    ):
        result = await normalize_for_pubmed(raw_term, drug_name=None)
        assert result == raw_term


# ── Organ specificity preserved ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw_term, llm_output, forbidden_exact",
    [
        ("non-small cell lung carcinoma", "lung cancer OR lung neoplasm", "cancer"),
        (
            "hepatocellular carcinoma",
            "liver cancer OR hepatocellular carcinoma",
            "cancer",
        ),
    ],
)
async def test_organ_specificity_preserved(raw_term, llm_output, forbidden_exact):
    """Result must retain organ context and not collapse to a bare blocklisted term."""
    with patch(
        "indication_scout.services.disease_normalizer.llm_normalize",
        new=AsyncMock(return_value=llm_output),
    ):
        result = await normalize_for_pubmed(raw_term, drug_name=None)
        terms = [t.strip().lower() for t in result.split("OR")]
        # Must not be a bare blocklisted term
        assert not (set(terms) <= BROADENING_BLOCKLIST)
        # Must not have collapsed to only the forbidden term
        assert forbidden_exact not in terms or len(terms) > 1


# ── Fallback blocklist guard ──────────────────────────────────────────────────


async def test_fallback_blocklist_rejects_overly_broad_broader_term():
    """If the fallback LLM generalizes to a blocklisted term, it is discarded."""
    call_count = 0

    async def mock_llm_normalize(term: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "renal tubular dysgenesis"  # first call: specific
        return "disease"  # second call (fallback): blocklisted

    with (
        patch(
            "indication_scout.services.disease_normalizer.llm_normalize",
            new=mock_llm_normalize,
        ),
        patch(
            "indication_scout.services.disease_normalizer.pubmed_count",
            new=AsyncMock(return_value=0),  # force fallback path
        ),
    ):
        result = await normalize_for_pubmed(
            "renal tubular dysgenesis", drug_name="metformin"
        )
        # Fallback was blocked; should keep the first LLM result
        assert result == "renal tubular dysgenesis"


async def test_fallback_accepted_when_not_blocklisted():
    """If the fallback term is specific enough, it is accepted."""
    call_count = 0

    async def mock_llm_normalize(term: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "renal tubular dysgenesis"
        return "kidney disease"  # not in blocklist

    with (
        patch(
            "indication_scout.services.disease_normalizer.llm_normalize",
            new=mock_llm_normalize,
        ),
        patch(
            "indication_scout.services.disease_normalizer.pubmed_count",
            new=AsyncMock(side_effect=[0, 10]),  # first: too few; second: enough
        ),
    ):
        result = await normalize_for_pubmed(
            "renal tubular dysgenesis", drug_name="metformin"
        )
        assert result == "kidney disease"
