"""Integration tests for the literature agent's tools.

Each tool is invoked directly via tool.ainvoke({...}) — the LLM agent loop is bypassed so we can
assert each tool's content/artifact in isolation. Tools share a closure-scoped store dict, so
each test that exercises a downstream tool first runs the upstream tools that populate the store.

Hits real Anthropic, PubMed, Open Targets, and ChEMBL APIs.
Uses the test database (scout_test) via db_session_truncating.
"""

import itertools
import logging
from datetime import date

from langchain_core.messages.tool import ToolCall

from indication_scout.agents.literature.literature_tools import build_literature_tools
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult, RetrievalService

logger = logging.getLogger(__name__)

# Counter to give each ToolCall a unique id within a test run
_tc_id = itertools.count()


def _tc(name: str, **args) -> ToolCall:
    """Build a ToolCall — required so .ainvoke() returns a ToolMessage with .artifact."""
    return ToolCall(name=name, args=args, id=f"tc{next(_tc_id)}", type="tool_call")


# ------------------------------------------------------------------
# Test target: Semaglutide + NASH, cutoff 2025-01-01
# Expected values reused from test_literature_agent.py (live run 2026-04-12)
# ------------------------------------------------------------------

_DRUG = "Semaglutide"
_DISEASE = "NASH"
_CUTOFF = date(2025, 1, 1)
_CHEMBL_ID = "CHEMBL2108724"

# PMIDs that must appear in fetch_and_cache output
_EXPECTED_PMIDS = {"39735270", "39412509"}

# Top-5 semantic search results (from test_literature_agent.py)
_EXPECTED_TOP5 = [
    ("37950798", "Potential New Therapeutic Implications of Semaglut"),
    ("37994050", "Lysophosphatidic acid receptor 1 antagonist (EPGN2"),
    ("38155202", "Semaglutide reduces tumor burden in the GAN diet-i"),
    ("36051145", "Semaglutide might be a key for breaking the viciou"),
    ("38464718", "Evolving role of semaglutide in NAFLD: in combinat"),
]
_EXPECTED_SUPPORTING_PMIDS = {"36051145", "38464718", "37950798", "38155202", "37994050"}

# --- Values to fill in from a live run -----------------------------
# build_drug_profile expected values (Semaglutide / CHEMBL2108724)
_EXPECTED_TARGET_GENE_SYMBOLS: list[str] = []        # TODO: fill in, e.g. ["GLP1R"]
_EXPECTED_MECHANISMS_OF_ACTION: list[str] = []       # TODO: fill in
_EXPECTED_ATC_CODES: list[str] = []                  # TODO: fill in, e.g. ["A10BJ06"]
_EXPECTED_ATC_DESCRIPTIONS: list[str] = []           # TODO: fill in
_EXPECTED_DRUG_TYPE: str = "Protein"                        # TODO: fill in, e.g. "Protein"

# Synthesize expected values
_EXPECTED_STRENGTH: str = "moderate"                     # confirmed by literature_agent test
_EXPECTED_MIN_STUDY_COUNT: int = 2                   # confirmed by literature_agent test
# -------------------------------------------------------------------


def _tool_map(tools: list) -> dict:
    return {t.name: t for t in tools}


def _build_tools(svc: RetrievalService, db):
    return build_literature_tools(svc, db, date_before=_CUTOFF)


async def test_build_drug_profile(db_session_truncating, test_cache_dir):
    """build_drug_profile resolves drug name → ChEMBL → DrugProfile with target/mechanism/ATC."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    msg = await tools["build_drug_profile"].ainvoke(_tc("build_drug_profile", drug_name=_DRUG))

    profile: DrugProfile = msg.artifact
    assert isinstance(profile, DrugProfile)
    assert profile.chembl_id == _CHEMBL_ID
    assert profile.drug_type == _EXPECTED_DRUG_TYPE

    # content string reflects actual counts
    assert msg.content == (
        f"Profile for {_DRUG} ({_CHEMBL_ID}): "
        f"{len(profile.target_gene_symbols)} targets, "
        f"{len(profile.mechanisms_of_action)} mechanisms"
    )


async def test_expand_search_terms(db_session_truncating, test_cache_dir):
    """expand_search_terms returns deduplicated PubMed queries that mention the drug and disease."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    # build_drug_profile populates the store; expand_search_terms reads from it
    await tools["build_drug_profile"].ainvoke(_tc("build_drug_profile", drug_name=_DRUG))
    msg = await tools["expand_search_terms"].ainvoke(
        _tc("expand_search_terms", drug_name=_DRUG, disease_name=_DISEASE)
    )

    queries: list[str] = msg.artifact
    assert isinstance(queries, list)
    assert all(isinstance(q, str) and q for q in queries)
    # case-insensitive dedup by retrieval.expand_search_terms
    assert len({q.lower().strip() for q in queries}) == len(queries)


    queries_lower = [q.lower() for q in queries]
    assert any("semaglutide" in q or "glp-1" in q or "glp1" in q for q in queries_lower)
    assert any(
        "nash" in q or "fatty liver" in q or "steatohepatitis" in q
        for q in queries_lower
    )

    assert msg.content == f"Generated {len(queries)} queries"


async def test_fetch_and_cache_without_queries(db_session_truncating, test_cache_dir):
    """fetch_and_cache short-circuits to an empty PMID list when expand_search_terms has not run."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    msg = await tools["fetch_and_cache"].ainvoke(_tc("fetch_and_cache", drug_name=_DRUG))

    assert msg.artifact == []
    assert msg.content == "No queries — call expand_search_terms first."


async def test_fetch_and_cache(db_session_truncating, test_cache_dir):
    """fetch_and_cache returns deduplicated PMIDs and persists abstracts to pgvector."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    await tools["build_drug_profile"].ainvoke(_tc("build_drug_profile", drug_name=_DRUG))
    await tools["expand_search_terms"].ainvoke(
        _tc("expand_search_terms", drug_name=_DRUG, disease_name=_DISEASE)
    )
    msg = await tools["fetch_and_cache"].ainvoke(_tc("fetch_and_cache", drug_name=_DRUG))

    pmids: list[str] = msg.artifact
    assert isinstance(pmids, list)
    assert all(isinstance(p, str) and p.isdigit() for p in pmids)
    # dedup
    assert len(pmids) == len(set(pmids))

    assert _EXPECTED_PMIDS.issubset(set(pmids))
    assert msg.content == f"Fetched {len(pmids)} PMIDs"


async def test_semantic_search_without_pmids(db_session_truncating, test_cache_dir):
    """semantic_search short-circuits to an empty list when fetch_and_cache has not run."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    msg = await tools["semantic_search"].ainvoke(
        _tc("semantic_search", drug_name=_DRUG, disease_name=_DISEASE)
    )

    assert msg.artifact == []
    assert msg.content == "No PMIDs — call fetch_and_cache first."


async def test_semantic_search(db_session_truncating, test_cache_dir):
    """semantic_search returns top-k AbstractResults sorted by descending similarity."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    await tools["build_drug_profile"].ainvoke(_tc("build_drug_profile", drug_name=_DRUG))
    await tools["expand_search_terms"].ainvoke(
        _tc("expand_search_terms", drug_name=_DRUG, disease_name=_DISEASE)
    )
    await tools["fetch_and_cache"].ainvoke(_tc("fetch_and_cache", drug_name=_DRUG))
    msg = await tools["semantic_search"].ainvoke(
        _tc("semantic_search", drug_name=_DRUG, disease_name=_DISEASE)
    )

    results: list[AbstractResult] = msg.artifact
    assert len(results) == 5
    assert all(isinstance(r, AbstractResult) for r in results)

    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)
    assert all(0.0 < s <= 1.0 for s in similarities)

    result_pmids = [r.pmid for r in results]
    for expected_pmid, expected_title_fragment in _EXPECTED_TOP5:
        assert expected_pmid in result_pmids, f"Expected PMID {expected_pmid} not in top-5"
        match = next(r for r in results if r.pmid == expected_pmid)
        assert expected_title_fragment in match.title
        assert isinstance(match.abstract, str) and len(match.abstract) > 0

    assert msg.content == f"Found {len(results)} abstracts (top sim: {results[0].similarity:.2f})"


async def test_synthesize(db_session_truncating, test_cache_dir):
    """synthesize turns the top abstracts into an EvidenceSummary with strength + supporting PMIDs."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    await tools["build_drug_profile"].ainvoke(_tc("build_drug_profile", drug_name=_DRUG))
    await tools["expand_search_terms"].ainvoke(
        _tc("expand_search_terms", drug_name=_DRUG, disease_name=_DISEASE)
    )
    await tools["fetch_and_cache"].ainvoke(_tc("fetch_and_cache", drug_name=_DRUG))
    await tools["semantic_search"].ainvoke(
        _tc("semantic_search", drug_name=_DRUG, disease_name=_DISEASE)
    )
    msg = await tools["synthesize"].ainvoke(
        _tc("synthesize", drug_name=_DRUG, disease_name=_DISEASE)
    )

    evidence: EvidenceSummary = msg.artifact
    assert isinstance(evidence, EvidenceSummary)
    assert evidence.strength == _EXPECTED_STRENGTH
    assert evidence.study_count >= _EXPECTED_MIN_STUDY_COUNT
    assert _EXPECTED_SUPPORTING_PMIDS.issubset(set(evidence.supporting_pmids))
    assert len(evidence.key_findings) >= 2

    assert msg.content == f"Evidence strength: {evidence.strength}"


async def test_finalize_analysis(db_session_truncating, test_cache_dir):
    """finalize_analysis echoes the summary string back as both content and artifact."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(_build_tools(svc, db_session_truncating))

    summary_text = "Semaglutide shows weak preclinical evidence for NASH in the retrieved abstracts."
    msg = await tools["finalize_analysis"].ainvoke(
        _tc("finalize_analysis", summary=summary_text)
    )

    assert msg.artifact == summary_text
    assert msg.content == "Analysis complete."
