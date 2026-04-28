"""Unit tests for supervisor_tools — header rendering and pure helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.supervisor.supervisor_tools import (
    _phase3_completion_years,
    build_supervisor_tools,
)
from indication_scout.models.model_clinical_trials import (
    CompletedTrialsResult,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)


# --- _phase3_completion_years ---


@pytest.mark.parametrize(
    "trials, expected",
    [
        # Empty input
        ([], []),
        # Single Phase 3 with full ISO date
        (
            [Trial(phase="Phase 3", completion_date="2019-06-15")],
            [2019],
        ),
        # Phase 3 with year-month only
        (
            [Trial(phase="Phase 3", completion_date="2017-03")],
            [2017],
        ),
        # Mixed phases — only Phase 3 counted
        (
            [
                Trial(phase="Phase 2", completion_date="2018-01-01"),
                Trial(phase="Phase 3", completion_date="2020-05-01"),
                Trial(phase="Phase 1/Phase 2", completion_date="2015-01-01"),
            ],
            [2020],
        ),
        # Multiple Phase 3 trials, dedup + sort ascending
        (
            [
                Trial(phase="Phase 3", completion_date="2019-12-31"),
                Trial(phase="Phase 3", completion_date="2014-08-01"),
                Trial(phase="Phase 3", completion_date="2019-01-15"),
                Trial(phase="Phase 3", completion_date="2017-07-04"),
            ],
            [2014, 2017, 2019],
        ),
    ],
)
def test_phase3_completion_years(trials, expected):
    assert _phase3_completion_years(trials) == expected


@pytest.mark.parametrize(
    "trials, expected",
    [
        # Phase 3 missing completion_date — skipped
        ([Trial(phase="Phase 3", completion_date=None)], []),
        # Phase 3 with malformed completion_date (no leading 4-digit year) — skipped
        ([Trial(phase="Phase 3", completion_date="N/A")], []),
        ([Trial(phase="Phase 3", completion_date="")], []),
        # Phase 3 with extra leading whitespace — skipped (we don't normalize)
        ([Trial(phase="Phase 3", completion_date=" 2019-01-01")], []),
    ],
)
def test_phase3_completion_years_skips_unusable(trials, expected):
    assert _phase3_completion_years(trials) == expected


# --- analyze_clinical_trials header rendering ---


async def _invoke_with_output(output: ClinicalTrialsOutput, disease: str = "nash"):
    """Build the supervisor tools with run_clinical_trials_agent stubbed and invoke
    analyze_clinical_trials, bypassing the candidate allowlist guard."""
    llm = MagicMock()
    svc = MagicMock()
    db = MagicMock()
    with patch(
        "indication_scout.agents.supervisor.supervisor_tools.run_clinical_trials_agent",
        new=AsyncMock(return_value=output),
    ), patch(
        "indication_scout.agents.supervisor.supervisor_tools.build_clinical_trials_agent",
        new=MagicMock(return_value=MagicMock()),
    ), patch(
        "indication_scout.agents.supervisor.supervisor_tools.build_mechanism_agent",
        new=MagicMock(return_value=MagicMock()),
    ):
        tools = build_supervisor_tools(llm=llm, svc=svc, db=db)
        by_name = {t.name: t for t in tools}
        tool = by_name["analyze_clinical_trials"]

        # Inject the disease into the tool's closure-scoped allowlist so the
        # candidate-guard short-circuit does not reject the call.
        coro_fn = tool.coroutine
        closure_vars = dict(zip(coro_fn.__code__.co_freevars, coro_fn.__closure__))
        closure_vars["allowed_diseases"].cell_contents[disease] = (disease, "competitor")

        return await tool.ainvoke({"drug_name": "semaglutide", "disease_name": disease})


async def test_analyze_clinical_trials_header_includes_phase3_years():
    """When Phase 3 trials have completion_date years, the header lists them."""
    output = ClinicalTrialsOutput(
        search=SearchTrialsResult(
            total_count=12,
            by_status={"RECRUITING": 3, "ACTIVE_NOT_RECRUITING": 1, "WITHDRAWN": 0},
            trials=[],
        ),
        completed=CompletedTrialsResult(
            total_count=5,
            phase3_count=3,
            trials=[
                Trial(phase="Phase 3", completion_date="2014-08-01"),
                Trial(phase="Phase 3", completion_date="2017-03-15"),
                Trial(phase="Phase 3", completion_date="2019-12-31"),
                Trial(phase="Phase 2", completion_date="2018-01-01"),
            ],
        ),
        terminated=TerminatedTrialsResult(total_count=2, trials=[]),
        approval=None,
        summary="",
    )

    summary = await _invoke_with_output(output)

    assert "3 Phase 3, completed 2014, 2017, 2019" in summary
    assert "[sample]" not in summary


async def test_analyze_clinical_trials_header_marks_sample_when_truncated():
    """When phase3_count > shown Phase 3 trials, the header marks the years as a sample."""
    output = ClinicalTrialsOutput(
        search=SearchTrialsResult(total_count=20, by_status={}, trials=[]),
        completed=CompletedTrialsResult(
            total_count=10,
            phase3_count=5,
            trials=[
                Trial(phase="Phase 3", completion_date="2018-01-01"),
                Trial(phase="Phase 3", completion_date="2020-06-01"),
            ],
        ),
        terminated=TerminatedTrialsResult(total_count=0, trials=[]),
        approval=None,
        summary="",
    )

    summary = await _invoke_with_output(output)

    assert "5 Phase 3, completed 2018, 2020 [sample]" in summary


async def test_analyze_clinical_trials_header_omits_years_when_none_usable():
    """When no Phase 3 trials have a usable completion_date, header keeps the old form."""
    output = ClinicalTrialsOutput(
        search=SearchTrialsResult(total_count=4, by_status={}, trials=[]),
        completed=CompletedTrialsResult(
            total_count=2,
            phase3_count=2,
            trials=[
                Trial(phase="Phase 3", completion_date=None),
                Trial(phase="Phase 3", completion_date="N/A"),
            ],
        ),
        terminated=TerminatedTrialsResult(total_count=0, trials=[]),
        approval=None,
        summary="",
    )

    summary = await _invoke_with_output(output)

    assert "(2 Phase 3)" in summary
    assert "completed 20" not in summary  # no year in the Phase 3 clause


# --- semaglutide × NAFLD regression: briefing surfaces MASH so the prompt's ---
# --- APPROVED-CANDIDATE SHORT-CIRCUIT case C can fire on NAFLD. -----------------


def _build_tools_with_drug_facts(
    drug_name: str,
    approved_indications: list[str],
    drug_aliases: list[str] | None = None,
):
    """Build supervisor tools and prepopulate drug_facts for `drug_name`.

    Returns (tools_by_name, drug_facts) so tests can call get_drug_briefing
    and verify the rendered output that the supervisor's APPROVED-CANDIDATE
    SHORT-CIRCUIT depends on.
    """
    llm = MagicMock()
    svc = MagicMock()
    db = MagicMock()
    with patch(
        "indication_scout.agents.supervisor.supervisor_tools.build_clinical_trials_agent",
        new=MagicMock(return_value=MagicMock()),
    ), patch(
        "indication_scout.agents.supervisor.supervisor_tools.build_mechanism_agent",
        new=MagicMock(return_value=MagicMock()),
    ):
        tools = build_supervisor_tools(llm=llm, svc=svc, db=db)

    by_name = {t.name: t for t in tools}

    # Reach drug_facts via _ensure_drug_entry's closure. None of the tools
    # close over drug_facts directly; they go through _ensure_drug_entry
    # (the writers) and _render_briefing (the reader), both of which do.
    fc = by_name["find_candidates"]
    fc_closure = dict(zip(fc.coroutine.__code__.co_freevars, fc.coroutine.__closure__))
    ensure_fn = fc_closure["_ensure_drug_entry"].cell_contents
    ensure_closure = dict(zip(ensure_fn.__code__.co_freevars, ensure_fn.__closure__))
    drug_facts = ensure_closure["drug_facts"].cell_contents

    drug_facts[drug_name.lower().strip()] = {
        "drug_name": drug_name,
        "drug_aliases": drug_aliases or [],
        "approved_indications": list(approved_indications),
        "mechanism_targets": [],
        "mechanism_disease_associations": [],
    }
    return by_name, drug_facts


def test_semaglutide_briefing_lists_mash_when_seeded():
    """The briefing the supervisor reads MUST include MASH for semaglutide.

    This is the regression that prevents the original failure: without MASH
    in the briefing, the supervisor cannot apply the APPROVED-CANDIDATE
    SHORT-CIRCUIT case C ("NAFLD is a SUPERSET of approved MASH") and would
    demote NAFLD as settled-unfavorable on the strength of completed Phase 3
    trials with no approval.
    """
    by_name, _ = _build_tools_with_drug_facts(
        drug_name="semaglutide",
        approved_indications=[
            "type 2 diabetes mellitus",
            "chronic weight management",
            "MASH",
        ],
        drug_aliases=["Ozempic", "Wegovy", "Rybelsus"],
    )

    briefing = by_name["get_drug_briefing"].invoke({"drug_name": "semaglutide"})

    assert "DRUG INTAKE: semaglutide" in briefing
    assert "Trade/generic names: Ozempic, Wegovy, Rybelsus" in briefing
    assert "FDA-approved indications:" in briefing
    assert "- MASH" in briefing
    assert "- type 2 diabetes mellitus" in briefing
    assert "- chronic weight management" in briefing


def test_briefing_handles_unknown_drug_gracefully():
    """get_drug_briefing on a drug with no facts should NOT crash, just return a
    well-formed empty-state briefing — the supervisor relies on this when it
    calls the tool before any sub-agent has populated drug_facts."""
    by_name, _ = _build_tools_with_drug_facts(
        drug_name="semaglutide",
        approved_indications=["MASH"],
    )

    briefing = by_name["get_drug_briefing"].invoke({"drug_name": "metformin"})

    assert "DRUG INTAKE: metformin" in briefing
    assert "no facts collected yet" in briefing
