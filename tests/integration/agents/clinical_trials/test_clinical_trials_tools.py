"""Integration test for the clinical trials tool layer.

Hits real NCBI (MeSH resolver) and real ClinicalTrials.gov.

Verifies that the tool layer resolves the indication to a MeSH preferred term
and forwards it to the client, which then uses CT.gov's server-side
`AREA[ConditionMeshTerm]` filter so that returned counts and trials are both
restricted to the systemic-hypertension descriptor (D006973).

Expected values verified live on 2026-04-25 with:
  drug=semaglutide, indication=hypertension, date_before=2025-01-01
"""

import logging
from datetime import date

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    SearchTrialsResult,
)

logger = logging.getLogger(__name__)

_CUTOFF = date(2025, 1, 1)
_HYPERTENSION_MESH_ID = "D006973"


async def test_search_trials_tool_uses_server_side_mesh_filter():
    """search_trials tool: resolves hypertension → ('D006973', 'Hypertension')
    and forwards the preferred term to the client, which builds
    `AREA[ConditionMeshTerm]"Hypertension"` server-side. The returned
    SearchTrialsResult.total_count and .trials are both filtered to that
    descriptor — no client-side _filter_by_mesh post-walk.
    """
    tools = build_clinical_trials_tools(date_before=_CUTOFF)
    search_trials = next(t for t in tools if t.name == "search_trials")

    msg = await search_trials.ainvoke(
        {
            "name": "search_trials",
            "args": {"drug": "semaglutide", "indication": "hypertension"},
            "id": "test-call-1",
            "type": "tool_call",
        }
    )

    # Content string format: "Search for {drug} × {indication}: {N} trials
    # (recruiting=..., active=..., withdrawn=..., unknown=...)" — no
    # truncation note when shown == total.
    assert "Search for semaglutide × hypertension: 2 trials" in msg.content
    assert "recruiting=2" in msg.content
    assert "active=0" in msg.content
    assert "withdrawn=0" in msg.content
    assert "unknown=0" in msg.content

    result = msg.artifact
    assert isinstance(result, SearchTrialsResult)
    assert result.total_count == 2
    assert result.by_status == {
        "RECRUITING": 2,
        "ACTIVE_NOT_RECRUITING": 0,
        "WITHDRAWN": 0,
        "UNKNOWN": 0,
    }
    assert len(result.trials) == 2

    nct_ids = {t.nct_id for t in result.trials}
    assert nct_ids == {"NCT06132477", "NCT05746039"}

    # Every returned trial carries D006973 in either mesh_conditions or
    # mesh_ancestors — confirming the server-side `AREA[ConditionMeshTerm]`
    # filter restricted results to the Hypertension descriptor subtree.
    for t in result.trials:
        cond_ids = {m.id for m in t.mesh_conditions}
        anc_ids = {m.id for m in t.mesh_ancestors}
        assert _HYPERTENSION_MESH_ID in (cond_ids | anc_ids), (
            f"{t.nct_id} missing {_HYPERTENSION_MESH_ID} in both "
            f"mesh_conditions={cond_ids} and mesh_ancestors={anc_ids}"
        )

    # Spot-check the trial with D006973 directly in mesh_conditions.
    direct = next(t for t in result.trials if t.nct_id == "NCT06132477")
    assert direct.phase == "Phase 4"
    assert direct.overall_status == "RECRUITING"
    assert direct.enrollment == 150
    assert direct.sponsor == "University of Missouri-Columbia"
    assert direct.start_date == "2024-02-01"
    assert _HYPERTENSION_MESH_ID in {m.id for m in direct.mesh_conditions}

    # Spot-check the trial with D006973 only in mesh_ancestors.
    ancestor_only = next(t for t in result.trials if t.nct_id == "NCT05746039")
    assert ancestor_only.phase == "Phase 1/Phase 2"
    assert ancestor_only.overall_status == "RECRUITING"
    assert ancestor_only.enrollment == 8
    assert ancestor_only.sponsor == "University of Pennsylvania"
    assert ancestor_only.start_date == "2024-01-29"
    assert _HYPERTENSION_MESH_ID not in {m.id for m in ancestor_only.mesh_conditions}
    assert _HYPERTENSION_MESH_ID in {m.id for m in ancestor_only.mesh_ancestors}


# ------------------------------------------------------------------
# check_fda_approval — hits real ChEMBL, openFDA, and Anthropic
# ------------------------------------------------------------------


# Verified live on 2026-04-25: ChEMBL resolves semaglutide to 9 drug names —
# the four brand/INN forms plus ChEMBL/Novo Nordisk research codes and the
# Spanish INN. Lock the full set so a regression in the resolver surfaces here.
_SEMAGLUTIDE_EXPECTED_NAMES = {
    "semaglutide",
    "semaglutida",
    "ozempic",
    "wegovy",
    "rybelsus",
    "nn-9535",
    "nn9535",
    "nnc 0113-0217",
    "nnc-0113-0217",
}


@pytest.mark.parametrize(
    "indication,is_approved,matched_indication,expected_content",
    [
        (
            "NASH",
            True,
            "NASH",
            "FDA approval check for semaglutide × NASH: APPROVED (checked 9 drug names)",
        ),
        (
            "alzheimer disease",
            False,
            None,
            "FDA approval check for semaglutide × alzheimer disease: "
            "not on FDA label (checked 9 drug names)",
        ),
    ],
)
async def test_check_fda_approval_semaglutide(
    indication, is_approved, matched_indication, expected_content
):
    """End-to-end: ChEMBL resolution, openFDA label fetch, and LLM approval
    extraction for semaglutide.

    - NASH: semaglutide IS FDA-approved → is_approved=True.
    - alzheimer disease: label exists, indication not on it → is_approved=False.
    matched_indication is the caller's input verbatim when is_approved is True,
    otherwise None.

    Numbers verified live on 2026-04-25: ChEMBL resolves to 9 drug names for
    semaglutide.
    """
    tools = build_clinical_trials_tools(date_before=None)
    check_fda_approval = next(t for t in tools if t.name == "check_fda_approval")

    msg = await check_fda_approval.ainvoke(
        {
            "name": "check_fda_approval",
            "args": {"drug": "semaglutide", "indication": indication},
            "id": f"it-fda-{indication}",
            "type": "tool_call",
        }
    )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is is_approved
    assert msg.artifact.label_found is True
    assert msg.artifact.matched_indication == matched_indication
    assert set(msg.artifact.drug_names_checked) == _SEMAGLUTIDE_EXPECTED_NAMES
    assert msg.content == expected_content
