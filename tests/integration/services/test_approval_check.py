"""Integration tests for approval_check service — hits real openFDA + LLM APIs."""

import pytest

from indication_scout.services.approval_check import get_fda_approved_diseases

async def test_random(test_cache_dir):
    result = await get_fda_approved_diseases(
        drug_names=["wegovy"],
        candidate_diseases=[
            "NASH"
        ],
        cache_dir=test_cache_dir,
    )

    assert "NASH" in result

async def test_ozempic_detects_type_2_diabetes(test_cache_dir):
    """Ozempic's label lists type 2 diabetes mellitus as an approved indication."""
    result = await get_fda_approved_diseases(
        drug_names=["ozempic"],
        candidate_diseases=[
            "type 2 diabetes mellitus",
            "alzheimer's disease",
        ],
        cache_dir=test_cache_dir,
    )

    assert "type 2 diabetes mellitus" in result
    assert "alzheimer's disease" not in result


async def test_xeljanz_detects_ulcerative_colitis(test_cache_dir):
    """Xeljanz labels list ulcerative colitis as an approved indication."""
    result = await get_fda_approved_diseases(
        drug_names=["xeljanz"],
        candidate_diseases=[
            "ulcerative colitis",
            "colorectal cancer",
        ],
        cache_dir=test_cache_dir,
    )

    assert "ulcerative colitis" in result
    assert "colorectal cancer" not in result


async def test_xeljanz_detects_rheumatoid_arthritis(test_cache_dir):
    """Xeljanz labels list rheumatoid arthritis as an approved indication."""
    result = await get_fda_approved_diseases(
        drug_names=["xeljanz"],
        candidate_diseases=[
            "rheumatoid arthritis",
            "lupus",
        ],
        cache_dir=test_cache_dir,
    )

    assert "rheumatoid arthritis" in result
    assert "lupus" not in result
