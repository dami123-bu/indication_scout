"""Unit tests for run_mechanism_agent output assembly.

The agent itself (create_react_agent) is not invoked — we mock agent.ainvoke
to return a fixed message history and verify that run_mechanism_agent correctly
extracts artifacts into a MechanismOutput.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from indication_scout.agents.mechanism.mechanism_agent import run_mechanism_agent
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.models.model_open_targets import Association, Pathway

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared test data
# ------------------------------------------------------------------

TARGET_MAP = {"PRKAA1": "ENSG00000132356", "PRKAA2": "ENSG00000162409"}

ASSOCIATIONS_PRKAA1 = [
    Association(
        disease_id="EFO_0000400",
        disease_name="type 2 diabetes mellitus",
        overall_score=0.85,
        datatype_scores={"genetic_association": 0.9, "literature": 0.7},
        therapeutic_areas=["metabolism"],
    )
]

PATHWAYS_PRKAA1 = [
    Pathway(
        pathway_id="R-HSA-9612973",
        pathway_name="AMPK inhibits chREBP transcriptional activation activity",
        top_level_pathway="Metabolism",
    )
]

NARRATIVE = "PRKAA1 shows strong genetic association with type 2 diabetes. AMPK pathway membership supports metabolic repurposing candidates."


def _make_agent(messages: list) -> MagicMock:
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": messages})
    return agent


def _tool_msg(name: str, artifact, tool_call_id: str | None = None) -> ToolMessage:
    return ToolMessage(
        content=f"result of {name}",
        artifact=artifact,
        name=name,
        tool_call_id=tool_call_id or f"id_{name}",
    )


# ------------------------------------------------------------------
# Full happy path
# ------------------------------------------------------------------


async def test_run_mechanism_agent_assembles_all_fields():
    """run_mechanism_agent extracts all tool artifacts into MechanismOutput."""
    messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        _tool_msg("get_drug_targets", TARGET_MAP),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("get_target_pathways", {"PRKAA1": PATHWAYS_PRKAA1}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert isinstance(output, MechanismOutput)
    assert output.drug_targets == TARGET_MAP
    assert output.associations == {"PRKAA1": ASSOCIATIONS_PRKAA1}
    assert output.pathways == {"PRKAA1": PATHWAYS_PRKAA1}
    assert output.summary == NARRATIVE


# ------------------------------------------------------------------
# Multiple targets accumulate correctly
# ------------------------------------------------------------------


async def test_run_mechanism_agent_multiple_targets_accumulate():
    """associations and pathways from multiple per-target calls are merged."""
    associations_prkaa2 = [
        Association(
            disease_id="EFO_0000270",
            disease_name="asthma",
            overall_score=0.4,
            datatype_scores={"literature": 0.4},
            therapeutic_areas=["respiratory"],
        )
    ]
    pathways_prkaa2 = [
        Pathway(
            pathway_id="R-HSA-380972",
            pathway_name="Energy dependent regulation of mTOR",
            top_level_pathway="Signal Transduction",
        )
    ]

    messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        _tool_msg("get_drug_targets", TARGET_MAP),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("get_target_associations", {"PRKAA2": associations_prkaa2}),
        _tool_msg("get_target_pathways", {"PRKAA1": PATHWAYS_PRKAA1}),
        _tool_msg("get_target_pathways", {"PRKAA2": pathways_prkaa2}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert output.associations == {
        "PRKAA1": ASSOCIATIONS_PRKAA1,
        "PRKAA2": associations_prkaa2,
    }
    assert output.pathways == {
        "PRKAA1": PATHWAYS_PRKAA1,
        "PRKAA2": pathways_prkaa2,
    }


# ------------------------------------------------------------------
# Missing tools leave defaults
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "omit_tool,field,default",
    [
        ("get_drug_targets", "drug_targets", {}),
        ("get_target_associations", "associations", {}),
        ("get_target_pathways", "pathways", {}),
        ("finalize_analysis", "summary", ""),
    ],
)
async def test_run_mechanism_agent_missing_tool_leaves_default(omit_tool, field, default):
    """When a tool's ToolMessage is absent, the corresponding output field stays at its default."""
    all_messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        _tool_msg("get_drug_targets", TARGET_MAP),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("get_target_pathways", {"PRKAA1": PATHWAYS_PRKAA1}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    messages = [m for m in all_messages if not (isinstance(m, ToolMessage) and m.name == omit_tool)]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert getattr(output, field) == default


# ------------------------------------------------------------------
# Non-ToolMessages are ignored
# ------------------------------------------------------------------


async def test_run_mechanism_agent_ignores_non_tool_messages():
    """AIMessages and HumanMessages in the history do not affect output assembly."""
    messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        AIMessage(content="I will start by fetching the targets."),
        _tool_msg("get_drug_targets", TARGET_MAP),
        AIMessage(content="Now fetching associations."),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("get_target_pathways", {"PRKAA1": PATHWAYS_PRKAA1}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert output.drug_targets == TARGET_MAP
    assert output.summary == NARRATIVE
