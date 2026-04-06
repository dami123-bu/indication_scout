"""Unit tests for literature_tools."""

import logging
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import ToolCall

from indication_scout.agents.literature.literature_tools import build_literature_tools
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult

logger = logging.getLogger(__name__)


def _drug_profile() -> DrugProfile:
    return DrugProfile(
        name="metformin",
        synonyms=["Glucophage"],
        target_gene_symbols=["PRKAA1"],
        mechanisms_of_action=["AMP-activated protein kinase activator"],
        atc_codes=["A10BA02"],
        atc_descriptions=["Biguanides"],
        drug_type="Small molecule",
    )


def _build_tools(svc):
    tools, store = build_literature_tools(svc=svc, db=MagicMock())
    tool_map = {t.name: t for t in tools}
    return tool_map, store


async def test_expand_search_terms_returns_queries():
    """expand_search_terms reads drug_profile from store and returns queries as artifact."""
    expected_queries = ["metformin colorectal cancer", "AMPK colon neoplasm"]
    svc = MagicMock()
    svc.expand_search_terms = AsyncMock(return_value=expected_queries)

    tool_map, store = _build_tools(svc)
    store["drug_profile"] = _drug_profile()

    msg = await tool_map["expand_search_terms"].ainvoke(
        ToolCall(
            name="expand_search_terms",
            args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
            id="tc1",
            type="tool_call",
        )
    )

    svc.expand_search_terms.assert_awaited_once_with(
        "metformin", "colorectal cancer", _drug_profile()
    )
    assert msg.artifact == ["metformin colorectal cancer", "AMPK colon neoplasm"]
    assert "2" in msg.content


async def test_semantic_search_passes_args_and_returns_results():
    """semantic_search reads pmids from store, calls svc, and returns AbstractResult list as artifact."""
    expected = [
        AbstractResult(
            pmid="111", title="Metformin and CRC", abstract="...", similarity=0.91
        ),
        AbstractResult(
            pmid="222", title="AMPK pathway", abstract="...", similarity=0.85
        ),
    ]
    svc = MagicMock()
    svc.semantic_search = AsyncMock(return_value=expected)
    db = MagicMock()

    tools, store = build_literature_tools(svc=svc, db=db, num_top_k=5)
    tool_map = {t.name: t for t in tools}
    store["pmids"] = ["111", "222"]

    msg = await tool_map["semantic_search"].ainvoke(
        ToolCall(
            name="semantic_search",
            args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
            id="tc2",
            type="tool_call",
        )
    )

    svc.semantic_search.assert_awaited_once_with(
        "colorectal cancer", "metformin", ["111", "222"], db, top_k=5
    )
    assert len(msg.artifact) == 2
    assert msg.artifact[0].pmid == "111"
    assert msg.artifact[0].title == "Metformin and CRC"
    assert msg.artifact[0].similarity == 0.91
    assert msg.artifact[1].pmid == "222"
    assert msg.artifact[1].title == "AMPK pathway"
    assert msg.artifact[1].similarity == 0.85
    assert "2" in msg.content


async def test_synthesize_returns_evidence_summary_as_artifact():
    """synthesize reads abstracts from store and returns EvidenceSummary as artifact."""
    expected = EvidenceSummary(
        strength="moderate",
        study_count=3,
        study_types=["RCT"],
        key_findings=["Metformin reduces tumor growth"],
        has_adverse_effects=False,
        supporting_pmids=["111", "222"],
        summary="Moderate evidence.",
    )
    abstracts = [
        AbstractResult(pmid="111", title="T", abstract="A", similarity=0.9),
    ]
    svc = MagicMock()
    svc.synthesize = AsyncMock(return_value=expected)

    tool_map, store = _build_tools(svc)
    store["semantic_search_results"] = abstracts

    msg = await tool_map["synthesize"].ainvoke(
        ToolCall(
            name="synthesize",
            args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
            id="tc3",
            type="tool_call",
        )
    )

    svc.synthesize.assert_awaited_once_with("metformin", "colorectal cancer", abstracts)
    assert isinstance(msg.artifact, EvidenceSummary)
    assert msg.artifact.strength == "moderate"
    assert msg.artifact.study_count == 3
    assert msg.artifact.study_types == ["RCT"]
    assert msg.artifact.key_findings == ["Metformin reduces tumor growth"]
    assert msg.artifact.has_adverse_effects is False
    assert msg.artifact.supporting_pmids == ["111", "222"]
    assert "moderate" in msg.content
