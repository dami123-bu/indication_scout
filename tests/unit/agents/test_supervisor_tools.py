"""Unit tests for supervisor_tools — briefing rendering."""

from unittest.mock import MagicMock, patch

from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools


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
