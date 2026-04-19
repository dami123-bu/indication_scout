"""Unit tests for supervisor_tools — FDA approval filtering in analyze_mechanism."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import ToolCall

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput, ShapedAssociation
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools

logger = logging.getLogger(__name__)

# -- Fixtures --

APPROVED_SHAPED = ShapedAssociation(
    target_symbol="PRKAA1",
    disease_name="type 2 diabetes mellitus",
    disease_id="EFO_0000400",
    overall_score=0.85,
    shape="neutral",
    rationale="test",
)

NON_APPROVED_SHAPED = ShapedAssociation(
    target_symbol="PRKAA1",
    disease_name="colorectal cancer",
    disease_id="EFO_0001360",
    overall_score=0.7,
    shape="neutral",
    rationale="test",
)

LOW_SCORE_SHAPED = ShapedAssociation(
    target_symbol="PRKAA1",
    disease_name="rare disorder",
    disease_id="EFO_0009999",
    overall_score=0.1,
    shape="neutral",
    rationale="test",
)


def _mechanism_output(shaped: list[ShapedAssociation]) -> MechanismOutput:
    return MechanismOutput(
        drug_targets={"PRKAA1": "ENSG00000132356"},
        shaped_associations=shaped,
        pathways={},
        summary="test",
    )


def _build_tools(tmp_path):
    """Build supervisor tools with a mocked RetrievalService and dummy LLM/db."""
    svc = MagicMock()
    svc.cache_dir = tmp_path
    llm = MagicMock()
    db = MagicMock()
    tools = build_supervisor_tools(llm=llm, svc=svc, db=db)
    return {t.name: t for t in tools}


# -- Tests --


async def test_analyze_mechanism_filters_fda_approved_diseases(tmp_path):
    """FDA-approved disease is excluded from the allowlist; non-approved is kept."""
    tool_map = _build_tools(tmp_path)

    mechanism_output = _mechanism_output([APPROVED_SHAPED, NON_APPROVED_SHAPED])

    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.run_mechanism_agent",
            new=AsyncMock(return_value=mechanism_output),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL1431"),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage"]),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.remove_approved_from_labels",
            new=AsyncMock(return_value={"colorectal cancer"}),
        ),
    ):
        msg = await tool_map["analyze_mechanism"].ainvoke(
            ToolCall(
                name="analyze_mechanism",
                args={"drug_name": "metformin"},
                id="tc_mech",
                type="tool_call",
            )
        )

    # Non-approved disease should appear in summary
    assert "colorectal cancer" in msg.content.lower()
    # Approved disease should NOT appear in the mechanism-added note
    assert "type 2 diabetes mellitus" not in msg.content.lower()
    # Exactly 1 disease added (colorectal cancer only)
    assert "1 mechanism-sourced diseases added" in msg.content


async def test_analyze_mechanism_keeps_non_approved_diseases(tmp_path):
    """When FDA returns all candidates as survivors, all qualifying diseases are added."""
    tool_map = _build_tools(tmp_path)

    mechanism_output = _mechanism_output([APPROVED_SHAPED, NON_APPROVED_SHAPED])

    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.run_mechanism_agent",
            new=AsyncMock(return_value=mechanism_output),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL1431"),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.get_all_drug_names",
            new=AsyncMock(return_value=["metformin", "glucophage"]),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.remove_approved_from_labels",
            new=AsyncMock(return_value={"type 2 diabetes mellitus", "colorectal cancer"}),
        ),
    ):
        msg = await tool_map["analyze_mechanism"].ainvoke(
            ToolCall(
                name="analyze_mechanism",
                args={"drug_name": "metformin"},
                id="tc_mech2",
                type="tool_call",
            )
        )

    # Both diseases should be added
    assert "2 mechanism-sourced diseases added" in msg.content


async def test_analyze_mechanism_skips_fda_check_when_no_drug_names(tmp_path):
    """When get_all_drug_names returns [], remove_approved_from_labels is not called."""
    tool_map = _build_tools(tmp_path)

    mechanism_output = _mechanism_output([APPROVED_SHAPED, NON_APPROVED_SHAPED])

    mock_fda = AsyncMock(return_value=set())

    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.run_mechanism_agent",
            new=AsyncMock(return_value=mechanism_output),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL1431"),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.get_all_drug_names",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.remove_approved_from_labels",
            new=mock_fda,
        ),
    ):
        msg = await tool_map["analyze_mechanism"].ainvoke(
            ToolCall(
                name="analyze_mechanism",
                args={"drug_name": "metformin"},
                id="tc_mech3",
                type="tool_call",
            )
        )

    mock_fda.assert_not_awaited()
    # Both diseases should be added (no FDA filtering)
    assert "2 mechanism-sourced diseases added" in msg.content
