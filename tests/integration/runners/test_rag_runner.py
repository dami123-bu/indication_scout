"""Integration tests for runners/rag_runner."""

import logging

import pytest

from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.runners.rag_runner import run_rag

logger = logging.getLogger(__name__)


async def test_run_rag_empagliflozin(db_session_truncating, test_cache_dir):
    results = await run_rag("empagliflozin", db_session_truncating, cache_dir=test_cache_dir)
    summaries = " ".join(r.summary.lower() for r in results.values())

    # Empagliflozin has broad positive evidence
    assert "support" in summaries or "evidence" in summaries

    # Basic structure checks
    for disease, summary in results.items():
        assert len(summary.summary) > 50
        assert summary.strength in ["strong", "moderate", "weak", "none"]
        assert isinstance(summary.has_adverse_effects, bool)


async def test_run_rag_thalidomide_mixed_signals(db_session_truncating, test_cache_dir):
    results = await run_rag("thalidomide", db_session_truncating, cache_dir=test_cache_dir)
    summaries = " ".join(r.summary.lower() for r in results.values())

    # Thalidomide should have some negative/cautionary signals
    assert "not support" in summaries or "no significant" in summaries or "contradicts" in summaries

    # Basic structure checks
    for disease, summary in results.items():
        assert len(summary.summary) > 50
        assert summary.strength in ["strong", "moderate", "weak", "none"]
        assert isinstance(summary.has_adverse_effects, bool)


