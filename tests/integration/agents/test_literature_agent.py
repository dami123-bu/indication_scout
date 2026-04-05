"""Integration tests for build_literature_graph.

These tests hit real Anthropic and Open Targets APIs.
They verify the agent calls expand_search_terms and produces structured output.
"""

import logging
from datetime import date

from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from indication_scout.agents.literature.literature_agent import build_literature_graph
from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 15


async def _run(drug: str, disease: str, date_before: date | None = None) -> object:
    settings = get_settings()
    engine = create_engine(settings.test_database_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    db.execute(text("TRUNCATE TABLE pubmed_abstracts"))
    db.commit()

    svc = RetrievalService(DEFAULT_CACHE_DIR)
    drug_profile = await svc.build_drug_profile(drug)
    graph = build_literature_graph(
        svc=svc, db=db, drug_profile=drug_profile, max_search_results=20, date_before=date_before
    )
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=f"Analyze {drug} in {disease}")],
            "drug_name": drug,
            "disease_name": disease,
            "date_before": date_before,
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return result["final_output"]


# ------------------------------------------------------------------
# semaglutide / NASH — date cutoff enforced
# ------------------------------------------------------------------


async def test_literature_agent_semaglutide_nash_date_cutoff():
    """Agent respects date_before cutoff: post-cutoff PMIDs must not appear in results."""
    cutoff = date(2009, 1, 1)
    output = await _run("Semaglutide", "NASH", date_before=cutoff)

    assert "21479465" in output.pmids
    assert "30510243" not in output.pmids
    assert len(output.summary) > 100

    assert len(output.semantic_search_results) > 0
    assert any(s["pmid"] == "16953843" for s in output.semantic_search_results)
    top = output.semantic_search_results[0]
    assert "pmid" in top
    assert "title" in top
    assert "abstract" in top
    assert "similarity" in top

    assert isinstance(output.evidence_summary, EvidenceSummary)
    assert output.evidence_summary.strength in {"strong", "moderate", "weak", "none"}
    assert output.evidence_summary.study_count >= 0
    assert isinstance(output.evidence_summary.key_findings, list)
    assert isinstance(output.evidence_summary.supporting_pmids, list)

    assert output.summary == output.evidence_summary.summary
    assert len(output.summary) > 0
