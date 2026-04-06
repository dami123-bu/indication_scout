"""Unit tests for literature agent tools."""

import logging
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import ToolCall

from indication_scout.agents.literature.literature_tools import build_literature_tools
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SEARCH_TERMS = [
    "metformin colorectal cancer",
    "metformin colon neoplasm AMPK",
    "biguanide colorectal carcinoma",
]

PMIDS = ["111", "222", "333"]

SEMANTIC_RESULTS = [
    AbstractResult(
        pmid="111", title="Metformin and CRC", abstract="Study A.", similarity=0.91
    ),
    AbstractResult(
        pmid="222", title="AMPK pathway", abstract="Study B.", similarity=0.85
    ),
]

EVIDENCE = EvidenceSummary(
    strength="moderate",
    study_count=2,
    study_types=["RCT"],
    key_findings=["Metformin reduces tumor growth"],
    has_adverse_effects=False,
    supporting_pmids=["111", "222"],
    summary="Moderate evidence supports metformin in colorectal cancer based on 2 RCTs.",
)


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


def _make_svc() -> MagicMock:
    svc = MagicMock()
    svc.build_drug_profile = AsyncMock(return_value=_drug_profile())
    svc.expand_search_terms = AsyncMock(return_value=SEARCH_TERMS)
    svc.fetch_and_cache = AsyncMock(return_value=PMIDS)
    svc.semantic_search = AsyncMock(return_value=SEMANTIC_RESULTS)
    svc.synthesize = AsyncMock(return_value=EVIDENCE)
    return svc


def _build_tools(svc):
    tools, store = build_literature_tools(svc=svc, db=MagicMock())
    tool_map = {t.name: t for t in tools}
    return tool_map, store


# ------------------------------------------------------------------
# expand_search_terms
# ------------------------------------------------------------------


async def test_expand_search_terms_calls_svc_and_returns_queries():
    """expand_search_terms calls svc.expand_search_terms with the drug profile from store."""
    svc = _make_svc()
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
    assert msg.artifact == SEARCH_TERMS
    assert "3 search queries" in msg.content


async def test_expand_search_terms_builds_profile_if_missing():
    """expand_search_terms calls svc.build_drug_profile when store has no drug_profile."""
    svc = _make_svc()
    tool_map, store = _build_tools(svc)
    # store is empty — no drug_profile

    await tool_map["expand_search_terms"].ainvoke(
        ToolCall(
            name="expand_search_terms",
            args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
            id="tc2",
            type="tool_call",
        )
    )

    svc.build_drug_profile.assert_awaited_once_with("metformin")


# ------------------------------------------------------------------
# fetch_and_cache
# ------------------------------------------------------------------


async def test_fetch_and_cache_calls_svc_with_queries_from_store():
    """fetch_and_cache reads expanded_search_results from store and passes them to svc."""
    svc = _make_svc()
    tool_map, store = _build_tools(svc)
    store["expanded_search_results"] = SEARCH_TERMS

    msg = await tool_map["fetch_and_cache"].ainvoke(
        ToolCall(
            name="fetch_and_cache",
            args={"drug_name": "metformin"},
            id="tc3",
            type="tool_call",
        )
    )

    svc.fetch_and_cache.assert_awaited_once()
    assert svc.fetch_and_cache.call_args.args[0] == SEARCH_TERMS
    assert msg.artifact == PMIDS
    assert "3 PMIDs" in msg.content


async def test_fetch_and_cache_returns_empty_when_no_queries():
    """fetch_and_cache returns early with empty list when store has no queries."""
    svc = _make_svc()
    tool_map, store = _build_tools(svc)
    # store is empty

    msg = await tool_map["fetch_and_cache"].ainvoke(
        ToolCall(
            name="fetch_and_cache",
            args={"drug_name": "metformin"},
            id="tc4",
            type="tool_call",
        )
    )

    svc.fetch_and_cache.assert_not_awaited()
    assert msg.artifact == []
    assert "expand_search_terms" in msg.content


# ------------------------------------------------------------------
# semantic_search
# ------------------------------------------------------------------


async def test_semantic_search_calls_svc_with_pmids_from_store():
    """semantic_search reads pmids from store and passes them to svc."""
    svc = _make_svc()
    tool_map, store = _build_tools(svc)
    store["pmids"] = PMIDS

    msg = await tool_map["semantic_search"].ainvoke(
        ToolCall(
            name="semantic_search",
            args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
            id="tc5",
            type="tool_call",
        )
    )

    svc.semantic_search.assert_awaited_once()
    assert svc.semantic_search.call_args.args[2] == PMIDS
    assert len(msg.artifact) == 2
    assert msg.artifact[0].pmid == "111"
    assert msg.artifact[0].title == "Metformin and CRC"
    assert msg.artifact[0].abstract == "Study A."
    assert msg.artifact[0].similarity == 0.91
    assert msg.artifact[1].pmid == "222"
    assert msg.artifact[1].title == "AMPK pathway"
    assert msg.artifact[1].abstract == "Study B."
    assert msg.artifact[1].similarity == 0.85
    assert "0.91" in msg.content


async def test_semantic_search_returns_empty_when_no_pmids():
    """semantic_search returns early with empty list when store has no pmids."""
    svc = _make_svc()
    tool_map, store = _build_tools(svc)
    # store is empty

    msg = await tool_map["semantic_search"].ainvoke(
        ToolCall(
            name="semantic_search",
            args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
            id="tc6",
            type="tool_call",
        )
    )

    svc.semantic_search.assert_not_awaited()
    assert msg.artifact == []
    assert "fetch_and_cache" in msg.content


# ------------------------------------------------------------------
# synthesize
# ------------------------------------------------------------------


async def test_synthesize_calls_svc_with_abstracts_from_store():
    """synthesize reads semantic_search_results from store and passes them to svc."""
    svc = _make_svc()
    tool_map, store = _build_tools(svc)
    store["semantic_search_results"] = SEMANTIC_RESULTS

    msg = await tool_map["synthesize"].ainvoke(
        ToolCall(
            name="synthesize",
            args={"drug_name": "metformin", "disease_name": "colorectal cancer"},
            id="tc7",
            type="tool_call",
        )
    )

    svc.synthesize.assert_awaited_once_with(
        "metformin", "colorectal cancer", SEMANTIC_RESULTS
    )
    assert isinstance(msg.artifact, EvidenceSummary)
    assert msg.artifact.strength == "moderate"
    assert msg.artifact.study_count == 2
    assert msg.artifact.study_types == ["RCT"]
    assert msg.artifact.key_findings == ["Metformin reduces tumor growth"]
    assert msg.artifact.has_adverse_effects is False
    assert msg.artifact.supporting_pmids == ["111", "222"]
    assert (
        msg.artifact.summary
        == "Moderate evidence supports metformin in colorectal cancer based on 2 RCTs."
    )
    assert "moderate" in msg.content
