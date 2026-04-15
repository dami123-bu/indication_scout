"""Unit tests for supervisor_tools — FDA approval filtering in analyze_mechanism."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import ToolCall

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools
from indication_scout.models.model_open_targets import Association, DrugData

logger = logging.getLogger(__name__)

# -- Fixtures --

APPROVED_DISEASE = Association(
    disease_id="EFO_0000400",
    disease_name="type 2 diabetes mellitus",
    overall_score=0.85,
    datatype_scores={"genetic_association": 0.9},
    therapeutic_areas=["metabolism"],
)

NON_APPROVED_DISEASE = Association(
    disease_id="EFO_0001360",
    disease_name="colorectal cancer",
    overall_score=0.7,
    datatype_scores={"genetic_association": 0.5},
    therapeutic_areas=["oncology"],
)

LOW_SCORE_DISEASE = Association(
    disease_id="EFO_0009999",
    disease_name="rare disorder",
    overall_score=0.1,
    datatype_scores={},
    therapeutic_areas=[],
)


def _mechanism_output(associations: dict[str, list[Association]]) -> MechanismOutput:
    return MechanismOutput(
        drug_targets={"PRKAA1": "ENSG00000132356"},
        associations=associations,
        shaped_associations=[],
        pathways={},
        summary="test",
    )


def _drug_data(trade_names: list[str]) -> DrugData:
    return DrugData(name="metformin", trade_names=trade_names)


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

    mechanism_output = _mechanism_output(
        {"PRKAA1": [APPROVED_DISEASE, NON_APPROVED_DISEASE]}
    )
    drug_data = _drug_data(trade_names=["glucophage"])

    mock_ot_client = AsyncMock()
    mock_ot_client.__aenter__ = AsyncMock(return_value=mock_ot_client)
    mock_ot_client.__aexit__ = AsyncMock(return_value=None)
    mock_ot_client.get_drug = AsyncMock(return_value=drug_data)

    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.run_mechanism_agent",
            new=AsyncMock(return_value=mechanism_output),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.OpenTargetsClient",
            return_value=mock_ot_client,
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.get_fda_approved_diseases",
            new=AsyncMock(return_value={"type 2 diabetes mellitus"}),
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
    """When FDA returns empty set, all qualifying diseases are added."""
    tool_map = _build_tools(tmp_path)

    mechanism_output = _mechanism_output(
        {"PRKAA1": [APPROVED_DISEASE, NON_APPROVED_DISEASE]}
    )
    drug_data = _drug_data(trade_names=["glucophage"])

    mock_ot_client = AsyncMock()
    mock_ot_client.__aenter__ = AsyncMock(return_value=mock_ot_client)
    mock_ot_client.__aexit__ = AsyncMock(return_value=None)
    mock_ot_client.get_drug = AsyncMock(return_value=drug_data)

    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.run_mechanism_agent",
            new=AsyncMock(return_value=mechanism_output),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.OpenTargetsClient",
            return_value=mock_ot_client,
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.get_fda_approved_diseases",
            new=AsyncMock(return_value=set()),
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


async def test_analyze_mechanism_skips_fda_check_when_no_trade_names(tmp_path):
    """When DrugData has empty trade names, get_fda_approved_diseases is not called."""
    tool_map = _build_tools(tmp_path)

    mechanism_output = _mechanism_output(
        {"PRKAA1": [APPROVED_DISEASE, NON_APPROVED_DISEASE]}
    )
    drug_data = _drug_data(trade_names=[])

    mock_ot_client = AsyncMock()
    mock_ot_client.__aenter__ = AsyncMock(return_value=mock_ot_client)
    mock_ot_client.__aexit__ = AsyncMock(return_value=None)
    mock_ot_client.get_drug = AsyncMock(return_value=drug_data)

    mock_fda = AsyncMock(return_value=set())

    with (
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.run_mechanism_agent",
            new=AsyncMock(return_value=mechanism_output),
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.OpenTargetsClient",
            return_value=mock_ot_client,
        ),
        patch(
            "indication_scout.agents.supervisor.supervisor_tools.get_fda_approved_diseases",
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
