"""Unit tests for build_literature_graph nodes."""

import logging
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage

from indication_scout.agents.literature.literature_agent import build_literature_graph
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SEARCH_TERMS = [
    "metformin colorectal cancer",
    "metformin colon neoplasm AMPK",
    "biguanide colorectal carcinoma",
]

PMIDS = ["111", "222", "333"]

SEMANTIC_RESULTS = [
    {
        "pmid": "111",
        "title": "Metformin and CRC",
        "abstract": "Study A.",
        "similarity": 0.91,
    },
    {
        "pmid": "222",
        "title": "AMPK pathway",
        "abstract": "Study B.",
        "similarity": 0.85,
    },
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
    svc.expand_search_terms = AsyncMock(return_value=SEARCH_TERMS)
    svc.fetch_and_cache = AsyncMock(return_value=PMIDS)
    svc.semantic_search = AsyncMock(return_value=SEMANTIC_RESULTS)
    svc.synthesize = AsyncMock(return_value=EVIDENCE)
    return svc


async def _run_graph(
    svc, drug: str = "metformin", disease: str = "colorectal cancer"
) -> dict:
    graph = build_literature_graph(
        svc=svc, db=MagicMock(), drug_profile=_drug_profile()
    )
    return await graph.ainvoke(
        {
            "messages": [HumanMessage(content=f"Analyze {drug} in {disease}")],
            "drug_name": drug,
            "disease_name": disease,
        }
    )


# ------------------------------------------------------------------
# expand_node
# ------------------------------------------------------------------


async def test_expand_node_stores_search_terms():
    """expand_node calls svc.expand_search_terms and stores results in state."""
    svc = _make_svc()
    result = await _run_graph(svc)

    svc.expand_search_terms.assert_awaited_once_with(
        "metformin", "colorectal cancer", _drug_profile()
    )
    output = result["final_output"]
    assert len(output.search_results) == 3
    assert output.search_results[0] == "metformin colorectal cancer"
    assert output.search_results[1] == "metformin colon neoplasm AMPK"
    assert output.search_results[2] == "biguanide colorectal carcinoma"


# ------------------------------------------------------------------
# fetch_node
# ------------------------------------------------------------------


async def test_fetch_node_stores_pmids():
    """fetch_node calls svc.fetch_and_cache with expanded search terms and stores pmids."""
    svc = _make_svc()
    result = await _run_graph(svc)

    svc.fetch_and_cache.assert_awaited_once()
    call_args = svc.fetch_and_cache.call_args
    assert call_args.args[0] == SEARCH_TERMS

    output = result["final_output"]
    assert len(output.pmids) == 3
    assert output.pmids[0] == "111"
    assert output.pmids[1] == "222"
    assert output.pmids[2] == "333"


# ------------------------------------------------------------------
# search_node
# ------------------------------------------------------------------


async def test_search_node_stores_semantic_results():
    """search_node calls svc.semantic_search with pmids from state and stores results."""
    svc = _make_svc()
    result = await _run_graph(svc)

    svc.semantic_search.assert_awaited_once()
    call_args = svc.semantic_search.call_args
    assert call_args.args[2] == PMIDS

    output = result["final_output"]
    assert len(output.semantic_search_results) == 2
    assert output.semantic_search_results[0]["pmid"] == "111"
    assert output.semantic_search_results[0]["title"] == "Metformin and CRC"
    assert output.semantic_search_results[0]["similarity"] == 0.91
    assert output.semantic_search_results[1]["pmid"] == "222"
    assert output.semantic_search_results[1]["title"] == "AMPK pathway"
    assert output.semantic_search_results[1]["similarity"] == 0.85


# ------------------------------------------------------------------
# synthesize_node
# ------------------------------------------------------------------


async def test_synthesize_node_stores_evidence_summary():
    """synthesize_node calls svc.synthesize with semantic results and stores EvidenceSummary."""
    svc = _make_svc()
    result = await _run_graph(svc)

    svc.synthesize.assert_awaited_once()
    call_args = svc.synthesize.call_args
    assert call_args.args[2] == SEMANTIC_RESULTS

    output = result["final_output"]
    assert isinstance(output.evidence_summary, EvidenceSummary)
    assert output.evidence_summary.strength == "moderate"
    assert output.evidence_summary.study_count == 2
    assert output.evidence_summary.study_types == ["RCT"]
    assert output.evidence_summary.key_findings == ["Metformin reduces tumor growth"]
    assert output.evidence_summary.has_adverse_effects is False
    assert output.evidence_summary.supporting_pmids == ["111", "222"]
    assert (
        output.summary
        == "Moderate evidence supports metformin in colorectal cancer based on 2 RCTs."
    )
