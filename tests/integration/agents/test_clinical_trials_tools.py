"""Integration test for the clinical trials tool layer.

Hits real NCBI (MeSH resolver) and real ClinicalTrials.gov.

Verifies that the tool layer resolves the indication to a MeSH D-number and
forwards it to the client so that the artifact only contains trials whose
mesh_conditions or mesh_ancestors include that D-number — i.e. Essie-driven
noise (glaucoma, portal, pulmonary hypertension) is dropped.

Expected values verified by a live run on 2026-04-19 with:
  drug=semaglutide, indication=hypertension, date_before=2025-01-01
"""

import logging
from datetime import date

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)

logger = logging.getLogger(__name__)

_CUTOFF = date(2025, 1, 1)
_HYPERTENSION_MESH_ID = "D006973"


async def test_search_trials_tool_forwards_resolved_mesh_id():
    """search_trials tool: resolves hypertension → D006973 and forwards it so
    the returned trials are restricted to the systemic-hypertension subtree.

    Baseline (without forwarding, live run 2026-04-19): 5 trials including
    NCT06792422, NCT06361823, NCT06027567 whose MeSH tags do NOT contain
    D006973. After forwarding, only the 2 D006973-tagged trials remain.
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

    assert msg.content == "Searched on semaglutide-hypertension and found 2 trials"

    trials = msg.artifact
    assert len(trials) == 2

    nct_ids = {t.nct_id for t in trials}
    assert nct_ids == {"NCT06132477", "NCT05746039"}

    # Every returned trial has D006973 in mesh_conditions or mesh_ancestors.
    for t in trials:
        cond_ids = {m.id for m in t.mesh_conditions}
        anc_ids = {m.id for m in t.mesh_ancestors}
        assert _HYPERTENSION_MESH_ID in (cond_ids | anc_ids), (
            f"{t.nct_id} missing {_HYPERTENSION_MESH_ID} in both "
            f"mesh_conditions={cond_ids} and mesh_ancestors={anc_ids}"
        )

    # Spot-check the D006973-in-mesh_conditions trial in detail.
    direct = next(t for t in trials if t.nct_id == "NCT06132477")
    assert direct.phase == "Phase 4"
    assert direct.overall_status == "RECRUITING"
    assert direct.enrollment == 150
    assert direct.sponsor == "University of Missouri-Columbia"
    assert direct.start_date == "2024-02-01"
    assert _HYPERTENSION_MESH_ID in {m.id for m in direct.mesh_conditions}

    # Spot-check the D006973-in-mesh_ancestors-only trial.
    ancestor_only = next(t for t in trials if t.nct_id == "NCT05746039")
    assert ancestor_only.phase == "Phase 1/Phase 2"
    assert ancestor_only.overall_status == "RECRUITING"
    assert ancestor_only.enrollment == 8
    assert ancestor_only.sponsor == "University of Pennsylvania"
    assert ancestor_only.start_date == "2024-01-29"
    assert _HYPERTENSION_MESH_ID not in {m.id for m in ancestor_only.mesh_conditions}
    assert _HYPERTENSION_MESH_ID in {m.id for m in ancestor_only.mesh_ancestors}
