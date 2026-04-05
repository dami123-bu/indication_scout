"""Unit tests for literature_tools."""

import logging
from unittest.mock import AsyncMock, MagicMock

from indication_scout.agents.literature.literature_tools import build_literature_tools
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary

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


def _get_tool(tools: list, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found in {[t.name for t in tools]}")


async def test_expand_search_terms_returns_queries():
    """expand_search_terms passes drug_profile from closure and returns queries as artifact."""
    expected_queries = ["metformin colorectal cancer", "AMPK colon neoplasm"]
    svc = MagicMock()
    svc.expand_search_terms = AsyncMock(return_value=expected_queries)
    profile = _drug_profile()

    tools = build_literature_tools(svc=svc, db=MagicMock(), drug_profile=profile)
    expand = _get_tool(tools, "expand_search_terms")

    content, artifact = await expand.ainvoke(
        {"drug_name": "metformin", "disease_name": "colorectal cancer"}
    )

    svc.expand_search_terms.assert_awaited_once_with(
        "metformin", "colorectal cancer", profile
    )
    assert artifact == ["metformin colorectal cancer", "AMPK colon neoplasm"]
    assert "2" in content


async def test_semantic_search_passes_args_and_returns_results():
    """semantic_search passes correct args to svc and returns abstract dicts as artifact."""
    expected = [
        {"pmid": "111", "title": "Metformin and CRC", "abstract": "...", "similarity": 0.91},
        {"pmid": "222", "title": "AMPK pathway", "abstract": "...", "similarity": 0.85},
    ]
    svc = MagicMock()
    svc.semantic_search = AsyncMock(return_value=expected)
    db = MagicMock()
    profile = _drug_profile()

    tools = build_literature_tools(svc=svc, db=db, drug_profile=profile, num_top_k=5)
    semantic = _get_tool(tools, "semantic_search")

    content, artifact = await semantic.ainvoke(
        {"drug_name": "metformin", "disease_name": "colorectal cancer", "pmids": ["111", "222"]}
    )

    svc.semantic_search.assert_awaited_once_with(
        "colorectal cancer", "metformin", ["111", "222"], db, 5
    )
    assert len(artifact) == 2
    assert artifact[0]["pmid"] == "111"
    assert artifact[0]["title"] == "Metformin and CRC"
    assert artifact[0]["similarity"] == 0.91
    assert artifact[1]["pmid"] == "222"
    assert artifact[1]["title"] == "AMPK pathway"
    assert artifact[1]["similarity"] == 0.85
    assert "2" in content


async def test_synthesize_returns_evidence_summary_as_artifact():
    """synthesize returns EvidenceSummary as artifact, bypassing JSON serialization."""
    expected = EvidenceSummary(
        strength="moderate",
        study_count=3,
        study_types=["RCT"],
        key_findings=["Metformin reduces tumor growth"],
        has_adverse_effects=False,
        supporting_pmids=["111", "222"],
    )
    svc = MagicMock()
    svc.synthesize = AsyncMock(return_value=expected)
    profile = _drug_profile()

    tools = build_literature_tools(svc=svc, db=MagicMock(), drug_profile=profile)
    synthesize = _get_tool(tools, "synthesize")

    abstracts = [{"pmid": "111", "title": "T", "abstract": "A", "similarity": 0.9}]
    content, artifact = await synthesize.ainvoke(
        {"drug_name": "metformin", "disease_name": "colorectal cancer", "abstracts": abstracts}
    )

    svc.synthesize.assert_awaited_once_with("metformin", "colorectal cancer", abstracts)
    assert isinstance(artifact, EvidenceSummary)
    assert artifact.strength == "moderate"
    assert artifact.study_count == 3
    assert artifact.study_types == ["RCT"]
    assert artifact.key_findings == ["Metformin reduces tumor growth"]
    assert artifact.has_adverse_effects is False
    assert artifact.supporting_pmids == ["111", "222"]
    assert "moderate" in content
    assert "3" in content
