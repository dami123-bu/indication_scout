"""Integration tests for the supervisor agent's tools.

Each tool is invoked directly via tool.ainvoke({...}) — the LLM agent loop is bypassed so we can
assert each tool's content/artifact in isolation. Tools share closure-scoped allowlist and
drug_facts dicts, so tests that exercise downstream tools first run the upstream tools that
populate them.

Hits real Anthropic, Open Targets, ChEMBL, openFDA, ClinicalTrials.gov, and PubMed APIs.
Uses the test database (scout_test) via db_session_truncating.
"""

import itertools
import logging
from datetime import date

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages.tool import ToolCall

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# Counter to give each ToolCall a unique id within a test run
_tc_id = itertools.count()


def _tc(name: str, **args) -> ToolCall:
    """Build a ToolCall — required so .ainvoke() returns a ToolMessage with .artifact."""
    return ToolCall(name=name, args=args, id=f"tc{next(_tc_id)}", type="tool_call")


def _tool_map(tools: list) -> dict:
    return {t.name: t for t in tools}


@pytest.fixture
def llm():
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)


# ------------------------------------------------------------------
# find_candidates — Open Targets + ChEMBL + openFDA seeding
#
# Reuses the metformin candidates verified live on 2026-04-08 in
# test_supervisor_agent.py.
# ------------------------------------------------------------------

_DRUG = "metformin"
_EXPECTED_CANDIDATES_SUBSET = {
    "polycystic ovary syndrome",
    "gestational diabetes",
    "metabolic syndrome",
    "insulin resistance",
    "prostate cancer",
}

# TODO: fill in expected metformin aliases from a live ChEMBL run
# _EXPECTED_ALIASES_SUBSET: set[str] = {"metformin", "glucophage"}

# TODO: fill in expected FDA-approved indications seeded for metformin from openFDA
# _EXPECTED_APPROVED_INDICATIONS_SUBSET: set[str] = {"type 2 diabetes mellitus"}


async def test_find_candidates_metformin(llm, db_session_truncating, test_cache_dir):
    """find_candidates returns Open Targets candidate diseases, populates the closure-scoped
    allowlist, and seeds drug aliases + FDA-approved indications into the briefing store.
    """
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(build_supervisor_tools(llm=llm, svc=svc, db=db_session_truncating))

    msg = await tools["find_candidates"].ainvoke(_tc("find_candidates", drug_name=_DRUG))

    diseases: list[str] = msg.artifact
    assert isinstance(diseases, list)
    assert all(isinstance(d, str) and d for d in diseases)
    assert _EXPECTED_CANDIDATES_SUBSET.issubset(set(diseases))

    # content string format from supervisor_tools.find_candidates
    assert msg.content.startswith(f"Found {len(diseases)} candidate diseases for {_DRUG}")

    # Briefing must reflect the seeded aliases + FDA-approved indications
    briefing = tools["get_drug_briefing"].invoke({"drug_name": _DRUG})
    assert briefing.startswith(f"DRUG INTAKE: {_DRUG}")
    assert "Trade/generic names:" in briefing
    assert "(not yet resolved)" not in briefing  # aliases were seeded
    # TODO: tighten once expected aliases are known
    # for alias in _EXPECTED_ALIASES_SUBSET:
    #     assert alias.lower() in briefing.lower()

    # FDA-approved indications were seeded from the label
    assert "FDA-approved indications:" in briefing
    assert "(none discovered in this run)" not in briefing
    # TODO: tighten once expected approved indications are known
    # for ind in _EXPECTED_APPROVED_INDICATIONS_SUBSET:
    #     assert ind.lower() in briefing.lower()

    # Mechanism hasn't run yet
    assert "(mechanism agent has not run)" in briefing


# ------------------------------------------------------------------
# Reject path — analyze_literature / analyze_clinical_trials reject
# diseases that are not in the allowlist. No external APIs hit.
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,empty_artifact_type",
    [
        ("analyze_literature", LiteratureOutput),
        ("analyze_clinical_trials", ClinicalTrialsOutput),
    ],
)
async def test_analyze_rejects_unlisted_disease(
    llm, db_session_truncating, test_cache_dir, tool_name, empty_artifact_type
):
    """analyze_literature / analyze_clinical_trials must reject any disease that wasn't
    surfaced by find_candidates or promoted by analyze_mechanism. Returns an empty artifact
    of the right type and a REJECTED: content message naming the tool.
    """
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(build_supervisor_tools(llm=llm, svc=svc, db=db_session_truncating))

    msg = await tools[tool_name].ainvoke(
        _tc(tool_name, drug_name=_DRUG, disease_name="not-a-real-disease")
    )

    assert msg.content.startswith("REJECTED: 'not-a-real-disease' is not in the allowed")
    assert tool_name in msg.content
    assert isinstance(msg.artifact, empty_artifact_type)
    # An empty default-constructed artifact has no sub-results populated
    assert msg.artifact == empty_artifact_type()


# ------------------------------------------------------------------
# finalize_supervisor — echoes the input summary as the artifact and
# returns the canonical completion content. No external APIs.
# ------------------------------------------------------------------


async def test_finalize_supervisor_echoes_summary(llm, db_session_truncating, test_cache_dir):
    """finalize_supervisor returns ('Supervisor analysis complete.', summary_input)."""
    svc = RetrievalService(test_cache_dir)
    tools = _tool_map(build_supervisor_tools(llm=llm, svc=svc, db=db_session_truncating))

    summary_text = (
        "Metformin shows the strongest repurposing signal in polycystic ovary syndrome, "
        "with supporting literature and active trials."
    )
    msg = await tools["finalize_supervisor"].ainvoke(
        _tc("finalize_supervisor", summary=summary_text)
    )

    assert msg.content == "Supervisor analysis complete."
    assert msg.artifact == summary_text


# ------------------------------------------------------------------
# Cutoff plumbing — analyze_clinical_trials must respect a date_before
# cutoff so the supervisor's investigation is reproducible across runs.
#
# NOTE: build_supervisor_tools does not currently accept a date_before
# parameter — sub-agents are built with no cutoff (see
# supervisor_tools.py:62). This test is written against the wiring as
# it should be: if the cutoff is plumbed through, every trial returned
# for a candidate disease must have a start_date strictly before the
# cutoff. Until that wiring lands, this test will fail at the
# build_supervisor_tools(... date_before=...) call.
# ------------------------------------------------------------------

_CUTOFF = date(2025, 1, 1)
_CUTOFF_DRUG = "semaglutide"
_CUTOFF_DISEASE = "hypertension"  # in metformin's candidate list? — use semaglutide's instead.


async def test_analyze_clinical_trials_respects_cutoff(
    llm, db_session_truncating, test_cache_dir
):
    """When build_supervisor_tools is given date_before=2025-01-01, analyze_clinical_trials
    must only return trials whose start_date precedes the cutoff. Mirrors the cutoff
    contract verified in tests/integration/agents/clinical_trials/test_clinical_trials_tools.py.
    """
    svc = RetrievalService(test_cache_dir)

    # TODO: once build_supervisor_tools accepts date_before, switch to:
    #   tools = _tool_map(
    #       build_supervisor_tools(
    #           llm=llm, svc=svc, db=db_session_truncating, date_before=_CUTOFF
    #       )
    #   )
    tools = _tool_map(build_supervisor_tools(llm=llm, svc=svc, db=db_session_truncating))

    # Populate the allowlist so analyze_clinical_trials doesn't reject the disease.
    await tools["find_candidates"].ainvoke(_tc("find_candidates", drug_name=_CUTOFF_DRUG))

    msg = await tools["analyze_clinical_trials"].ainvoke(
        _tc(
            "analyze_clinical_trials",
            drug_name=_CUTOFF_DRUG,
            disease_name=_CUTOFF_DISEASE,
        )
    )

    output: ClinicalTrialsOutput = msg.artifact
    assert isinstance(output, ClinicalTrialsOutput)

    # Every trial in every scope (search / completed / terminated) must
    # have started before the cutoff.
    scopes = []
    if output.search is not None:
        scopes.append(("search", output.search.trials))
    if output.completed is not None:
        scopes.append(("completed", output.completed.trials))
    if output.terminated is not None:
        scopes.append(("terminated", output.terminated.trials))

    for scope_name, trials in scopes:
        for t in trials:
            assert t.start_date is not None, (
                f"{scope_name} trial {t.nct_id} has no start_date — "
                f"cannot verify cutoff compliance"
            )
            assert t.start_date < _CUTOFF.isoformat(), (
                f"{scope_name} trial {t.nct_id} start_date={t.start_date} "
                f"is not before cutoff {_CUTOFF.isoformat()}"
            )
