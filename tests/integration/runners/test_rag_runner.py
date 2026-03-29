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

    # Thalidomide has legitimate therapeutic uses (MDS, lymphoma, myeloma)
    # but is well-known for adverse effects — at least some indications should flag this.
    adverse_count = sum(1 for s in results.values() if s.has_adverse_effects)
    assert adverse_count > 0, "Expected at least one indication to flag adverse effects for thalidomide"

    # Not everything should come back as "strong" — myelofibrosis RCT failed,
    # brain cancer showed limited single-agent activity.
    strengths = [s.strength for s in results.values()]
    assert "weak" in strengths or "none" in strengths, (
        f"Expected at least one weak/none signal for thalidomide, got: {strengths}"
    )

    # Summaries should contain cautionary language reflecting negative efficacy signals
    summaries = " ".join(r.summary.lower() for r in results.values())
    cautionary_terms = ["fail", "limited", "inconsistent", "low", "modest", "not support", "no significant"]
    matches = [t for t in cautionary_terms if t in summaries]
    assert len(matches) >= 2, (
        f"Expected at least 2 cautionary terms in thalidomide summaries, found: {matches}"
    )

    for disease, summary in results.items():
        assert len(summary.summary) > 50
        assert summary.strength in ["strong", "moderate", "weak", "none"]
        assert isinstance(summary.has_adverse_effects, bool)


