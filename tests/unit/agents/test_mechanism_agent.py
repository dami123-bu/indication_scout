"""Unit tests for run_mechanism_agent output assembly.

The agent itself (create_react_agent) is not invoked — we mock agent.ainvoke
to return a fixed message history and verify that run_mechanism_agent correctly
extracts artifacts and the narrative summary into a MechanismOutput.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from indication_scout.agents.mechanism.mechanism_agent import run_mechanism_agent
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.models.model_drug_profile import DrugProfile

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared test data
# ------------------------------------------------------------------

COMPETITORS_RAW = {
    "diseases": {
        "type 2 diabetes": ["glipizide", "glyburide"],
        "obesity": ["orlistat"],
    },
    "drug_indications": ["type 2 diabetes mellitus"],
}

DRUG_PROFILE = DrugProfile(
    name="metformin",
    synonyms=["Glucophage", "dimethylbiguanide"],
    target_gene_symbols=["PRKAA1", "PRKAA2"],
    mechanisms_of_action=["AMP-activated protein kinase activator"],
    atc_codes=["A10BA02"],
    atc_descriptions=["Biguanides", "Metformin"],
    drug_type="Small molecule",
)

SEARCH_QUERIES = {
    "colorectal cancer": [
        "metformin colorectal cancer AMPK",
        "metformin colon neoplasm randomized",
    ],
}

NARRATIVE = "Metformin shares targets with competitors in diabetes and obesity. AMPK activation may be relevant to colorectal cancer repurposing."


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


def _ai_msg_with_tool_call(tool_call_id: str, tool_name: str, args: dict) -> AIMessage:
    msg = AIMessage(content="")
    msg.tool_calls = [{"id": tool_call_id, "name": tool_name, "args": args}]
    return msg


# ------------------------------------------------------------------
# Full happy path
# ------------------------------------------------------------------


async def test_run_mechanism_agent_assembles_all_fields():
    """run_mechanism_agent extracts all tool artifacts and narrative into MechanismOutput."""
    expand_call_id = "id_expand_colorectal"
    messages = [
        HumanMessage(content="Analyze metformin"),
        _tool_msg("get_drug_competitors", COMPETITORS_RAW),
        _tool_msg("build_drug_profile", DRUG_PROFILE),
        _ai_msg_with_tool_call(
            expand_call_id,
            "expand_search_terms",
            {"drug_name": "metformin", "disease": "colorectal cancer"},
        ),
        _tool_msg(
            "expand_search_terms",
            SEARCH_QUERIES["colorectal cancer"],
            tool_call_id=expand_call_id,
        ),
        AIMessage(content=NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert isinstance(output, MechanismOutput)
    assert output.competitors == {
        "type 2 diabetes": ["glipizide", "glyburide"],
        "obesity": ["orlistat"],
    }
    assert output.drug_indications == ["type 2 diabetes mellitus"]
    assert isinstance(output.drug_profile, DrugProfile)
    assert output.drug_profile.name == "metformin"
    assert output.drug_profile.synonyms == ["Glucophage", "dimethylbiguanide"]
    assert output.drug_profile.target_gene_symbols == ["PRKAA1", "PRKAA2"]
    assert output.drug_profile.mechanisms_of_action == ["AMP-activated protein kinase activator"]
    assert output.drug_profile.atc_codes == ["A10BA02"]
    assert output.drug_profile.atc_descriptions == ["Biguanides", "Metformin"]
    assert output.drug_profile.drug_type == "Small molecule"
    assert output.search_queries == {
        "colorectal cancer": [
            "metformin colorectal cancer AMPK",
            "metformin colon neoplasm randomized",
        ]
    }
    assert output.summary == NARRATIVE


# ------------------------------------------------------------------
# Summary extraction: last AIMessage without tool_calls wins
# ------------------------------------------------------------------


async def test_run_mechanism_agent_picks_last_ai_message_without_tool_calls():
    """The narrative summary is taken from the last AIMessage that has no tool_calls."""
    first_narrative = "First draft summary — should be ignored."
    final_narrative = "Final summary — this is the one."
    messages = [
        HumanMessage(content="Analyze metformin"),
        _tool_msg("get_drug_competitors", COMPETITORS_RAW),
        AIMessage(content=first_narrative),
        _tool_msg("build_drug_profile", DRUG_PROFILE),
        AIMessage(content=final_narrative),
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert output.summary == final_narrative


# ------------------------------------------------------------------
# Missing tools leave defaults
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "omit_tool,missing_field,default_value",
    [
        ("get_drug_competitors", "competitors", {}),
        ("get_drug_competitors", "drug_indications", []),
        ("build_drug_profile", "drug_profile", None),
        ("expand_search_terms", "search_queries", {}),
    ],
)
async def test_run_mechanism_agent_missing_tool_leaves_default(
    omit_tool, missing_field, default_value
):
    """When a tool's ToolMessage is absent, the corresponding output field stays at its default."""
    expand_call_id = "id_expand_colorectal"
    all_messages = [
        HumanMessage(content="Analyze metformin"),
        _tool_msg("get_drug_competitors", COMPETITORS_RAW),
        _tool_msg("build_drug_profile", DRUG_PROFILE),
        _ai_msg_with_tool_call(
            expand_call_id,
            "expand_search_terms",
            {"drug_name": "metformin", "disease": "colorectal cancer"},
        ),
        _tool_msg(
            "expand_search_terms",
            SEARCH_QUERIES["colorectal cancer"],
            tool_call_id=expand_call_id,
        ),
        AIMessage(content=NARRATIVE),
    ]
    messages = [m for m in all_messages if not (isinstance(m, ToolMessage) and m.name == omit_tool)]
    # Also remove the AIMessage with tool_calls for expand_search_terms when omitting it
    if omit_tool == "expand_search_terms":
        messages = [
            m for m in messages
            if not (isinstance(m, AIMessage) and getattr(m, "tool_calls", []))
        ]

    agent = _make_agent(messages)
    output = await run_mechanism_agent(agent, "metformin")

    assert getattr(output, missing_field) == default_value


# ------------------------------------------------------------------
# Multiple expand_search_terms calls (one per disease)
# ------------------------------------------------------------------


async def test_run_mechanism_agent_multiple_expand_calls():
    """search_queries accumulates results from multiple expand_search_terms calls."""
    id_diabetes = "id_expand_diabetes"
    id_obesity = "id_expand_obesity"
    queries_diabetes = ["metformin type 2 diabetes AMPK", "biguanide insulin resistance"]
    queries_obesity = ["metformin obesity adipose AMPK", "metformin weight loss RCT"]

    messages = [
        HumanMessage(content="Analyze metformin"),
        _tool_msg("get_drug_competitors", COMPETITORS_RAW),
        _tool_msg("build_drug_profile", DRUG_PROFILE),
        _ai_msg_with_tool_call(
            id_diabetes,
            "expand_search_terms",
            {"drug_name": "metformin", "disease": "type 2 diabetes"},
        ),
        _tool_msg("expand_search_terms", queries_diabetes, tool_call_id=id_diabetes),
        _ai_msg_with_tool_call(
            id_obesity,
            "expand_search_terms",
            {"drug_name": "metformin", "disease": "obesity"},
        ),
        _tool_msg("expand_search_terms", queries_obesity, tool_call_id=id_obesity),
        AIMessage(content=NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert output.search_queries == {
        "type 2 diabetes": queries_diabetes,
        "obesity": queries_obesity,
    }
