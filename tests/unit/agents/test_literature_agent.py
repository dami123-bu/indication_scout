"""Unit tests for build_literature_graph nodes."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from indication_scout.agents.literature.literature_agent import build_literature_graph
from indication_scout.models.model_drug_profile import DrugProfile

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SEARCH_TERMS = [
    "metformin colorectal cancer",
    "metformin colon neoplasm AMPK",
    "biguanide colorectal carcinoma",
]

SUMMARY_TEXT = "Generated 3 diverse PubMed queries for metformin in colorectal cancer."


def _drug_profile() -> DrugProfile:
    return DrugProfile(
        name="metformin",
        synonyms=["Glucophage"],
        target_gene_symbols=["PRKAA1"],
        mechanisms_of_action=["AMP-activated protein kinase activator"],
        atc_codes=["A10BA02"],
        atc_descriptions=["Biguanides"],
        drug_type="Small molecule",
    )


def _make_llm(turn_responses: list) -> MagicMock:
    """Build a mock LLM that returns pre-scripted responses in sequence."""
    mock_llm = MagicMock()
    bound = AsyncMock()
    bound.ainvoke = AsyncMock(side_effect=turn_responses)
    mock_llm.bind_tools = MagicMock(return_value=bound)
    return mock_llm


def _tool_call(name: str, args: dict, call_id: str) -> ToolCall:
    return ToolCall(name=name, args=args, id=call_id)


def _ai_with_tool_calls(tool_calls: list[ToolCall]) -> AIMessage:
    return AIMessage(content="", tool_calls=tool_calls)


def _ai_final(content: str) -> AIMessage:
    return AIMessage(content=content)


def _tool_response(tool_call_id: str, name: str, data) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(data),
        tool_call_id=tool_call_id,
        name=name,
    )


async def _run_graph(llm, drug: str, disease: str) -> dict:
    svc = MagicMock()
    graph = build_literature_graph(llm, svc=svc, drug_profile=_drug_profile())
    return await graph.ainvoke(
        {
            "messages": [HumanMessage(content=f"Analyze {drug} in {disease}")],
            "drug_name": drug,
            "disease_name": disease,
        },
        config={"recursion_limit": 10},
    )


# ------------------------------------------------------------------
# tools_node: expand_search_terms parsed into state
# ------------------------------------------------------------------


async def test_tools_node_parses_search_results_into_state():
    """tools_node correctly parses expand_search_terms output into state.search_results."""
    from unittest.mock import patch, AsyncMock as AM

    llm = _make_llm([
        _ai_with_tool_calls([_tool_call("expand_search_terms", {"drug_name": "metformin", "disease_name": "colorectal cancer"}, "tc1")]),
        _ai_final(SUMMARY_TEXT),
    ])

    tool_responses = [
        {"messages": [_tool_response("tc1", "expand_search_terms", SEARCH_TERMS)]},
    ]
    mock_tool_node = AM()
    mock_tool_node.ainvoke = AM(side_effect=tool_responses)

    with patch(
        "indication_scout.agents.literature.literature_agent.ToolNode",
        return_value=mock_tool_node,
    ):
        result = await _run_graph(llm, "metformin", "colorectal cancer")

    output = result["final_output"]
    assert output is not None
    assert len(output.search_results) == 3
    assert output.search_results[0] == "metformin colorectal cancer"
    assert output.search_results[1] == "metformin colon neoplasm AMPK"
    assert output.search_results[2] == "biguanide colorectal carcinoma"


# ------------------------------------------------------------------
# assemble_node: summary extracted from final AIMessage
# ------------------------------------------------------------------


async def test_assemble_node_extracts_summary_from_final_ai_message():
    """assemble_node picks up the final AIMessage content as the summary."""
    from unittest.mock import patch, AsyncMock as AM

    llm = _make_llm([
        _ai_with_tool_calls([_tool_call("expand_search_terms", {"drug_name": "metformin", "disease_name": "colorectal cancer"}, "tc1")]),
        _ai_final(SUMMARY_TEXT),
    ])

    tool_responses = [
        {"messages": [_tool_response("tc1", "expand_search_terms", SEARCH_TERMS)]},
    ]
    mock_tool_node = AM()
    mock_tool_node.ainvoke = AM(side_effect=tool_responses)

    with patch(
        "indication_scout.agents.literature.literature_agent.ToolNode",
        return_value=mock_tool_node,
    ):
        result = await _run_graph(llm, "metformin", "colorectal cancer")

    output = result["final_output"]
    assert output.summary == SUMMARY_TEXT
