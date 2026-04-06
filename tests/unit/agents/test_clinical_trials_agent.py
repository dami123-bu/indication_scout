"""Unit tests for build_clinical_trials_graph nodes."""

import json
import logging
from datetime import date
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from for_me.clinical_trials.v3_langgraph.clinical_trials_agent import (
    build_clinical_trials_graph,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

WHITESPACE_DATA = {
    "is_whitespace": True,
    "no_data": False,
    "exact_match_count": 0,
    "drug_only_trials": 120,
    "indication_only_trials": 250,
    "indication_drugs": [
        {
            "nct_id": "NCT00652457",
            "drug_name": "Memantine",
            "indication": "Huntington's Disease",
            "phase": "Phase 4",
            "status": "COMPLETED",
        }
    ],
}

TERMINATED_DATA = [
    {
        "nct_id": "NCT01234567",
        "drug_name": "SomeDrug",
        "indication": "Huntington's Disease",
        "phase": "Phase 2",
        "why_stopped": "Lack of efficacy",
        "stop_category": "efficacy",
    }
]

LANDSCAPE_DATA = {
    "total_trial_count": 30,
    "competitors": [],
    "phase_distribution": {"Phase 2": 10, "Phase 3": 5},
    "recent_starts": [],
}

TRIALS_DATA = [
    {
        "nct_id": "NCT02970942",
        "title": "Riluzole ALS Trial",
        "brief_summary": "Testing riluzole in ALS",
        "phase": "Phase 2",
        "overall_status": "COMPLETED",
        "why_stopped": None,
        "indications": ["ALS"],
        "interventions": [
            {
                "intervention_type": "Drug",
                "intervention_name": "Riluzole",
                "description": "Daily oral dose",
            }
        ],
        "sponsor": "Sponsor Inc",
        "enrollment": 100,
        "start_date": "2015-01-01",
        "completion_date": "2018-01-01",
        "primary_outcomes": [{"measure": "Survival", "time_frame": "18 months"}],
        "references": ["12345678"],
    }
]

SUMMARY_TEXT = (
    "No trials exist for this drug-disease pair. One prior efficacy failure found."
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


async def _run_graph(llm, drug: str, disease: str, date_before=None) -> dict:
    graph = build_clinical_trials_graph(
        llm, max_search_results=50, date_before=date_before
    )
    return await graph.ainvoke(
        {
            "messages": [HumanMessage(content=f"Analyze {drug} in {disease}")],
            "drug_name": drug,
            "disease_name": disease,
            "date_before": date_before,
        },
        config={"recursion_limit": 15},
    )


# ------------------------------------------------------------------
# Whitespace path: detect_whitespace → get_terminated + get_landscape → summary
# ------------------------------------------------------------------


async def test_whitespace_path_tools_node_parses_state_correctly():
    """tools_node correctly parses whitespace, terminated, and landscape into state."""
    llm = _make_llm(
        [
            # Turn 1: call detect_whitespace
            _ai_with_tool_calls(
                [
                    _tool_call(
                        "detect_whitespace",
                        {"drug": "somedrug", "indication": "huntingtons"},
                        "tc1",
                    )
                ]
            ),
            # Turn 2: call get_terminated + get_landscape in one batch
            _ai_with_tool_calls(
                [
                    _tool_call(
                        "get_terminated",
                        {"drug": "somedrug", "indication": "huntingtons"},
                        "tc2",
                    ),
                    _tool_call("get_landscape", {"indication": "huntingtons"}, "tc3"),
                ]
            ),
            # Turn 3: final summary
            _ai_final(SUMMARY_TEXT),
        ]
    )

    # Patch ToolNode to return pre-scripted ToolMessages without hitting real APIs
    from unittest.mock import patch, AsyncMock as AM

    tool_responses_by_turn = [
        # After turn 1 tool calls
        {"messages": [_tool_response("tc1", "detect_whitespace", WHITESPACE_DATA)]},
        # After turn 2 tool calls
        {
            "messages": [
                _tool_response("tc2", "get_terminated", TERMINATED_DATA),
                _tool_response("tc3", "get_landscape", LANDSCAPE_DATA),
            ]
        },
    ]
    mock_tool_node_instance = AM()
    mock_tool_node_instance.ainvoke = AM(side_effect=tool_responses_by_turn)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_agent.ToolNode",
        return_value=mock_tool_node_instance,
    ):
        result = await _run_graph(llm, "somedrug", "huntingtons")

    output = result["final_output"]

    # Whitespace
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is True
    assert output.whitespace.no_data is False
    assert output.whitespace.exact_match_count == 0
    assert output.whitespace.drug_only_trials == 120
    assert output.whitespace.indication_only_trials == 250
    assert len(output.whitespace.indication_drugs) == 1
    assert output.whitespace.indication_drugs[0].nct_id == "NCT00652457"
    assert output.whitespace.indication_drugs[0].drug_name == "Memantine"
    assert output.whitespace.indication_drugs[0].phase == "Phase 4"
    assert output.whitespace.indication_drugs[0].status == "COMPLETED"

    # Terminated
    assert len(output.terminated) == 1
    assert output.terminated[0].nct_id == "NCT01234567"
    assert output.terminated[0].drug_name == "SomeDrug"
    assert output.terminated[0].phase == "Phase 2"
    assert output.terminated[0].why_stopped == "Lack of efficacy"
    assert output.terminated[0].stop_category == "efficacy"

    # Landscape
    assert output.landscape is not None
    assert output.landscape.total_trial_count == 30
    assert output.landscape.competitors == []
    assert output.landscape.phase_distribution == {"Phase 2": 10, "Phase 3": 5}
    assert output.landscape.recent_starts == []

    # No trials
    assert output.trials == []


# ------------------------------------------------------------------
# Active trials path: detect_whitespace → search_trials + get_landscape → summary
# ------------------------------------------------------------------


async def test_active_trials_path_tools_node_parses_state_correctly():
    """tools_node correctly parses trials and landscape when trials exist."""
    active_whitespace = {
        "is_whitespace": False,
        "no_data": False,
        "exact_match_count": 5,
        "drug_only_trials": 200,
        "indication_only_trials": 1000,
        "indication_drugs": [],
    }

    llm = _make_llm(
        [
            _ai_with_tool_calls(
                [
                    _tool_call(
                        "detect_whitespace",
                        {"drug": "riluzole", "indication": "als"},
                        "tc1",
                    )
                ]
            ),
            _ai_with_tool_calls(
                [
                    _tool_call(
                        "search_trials",
                        {"drug": "riluzole", "indication": "als"},
                        "tc2",
                    ),
                    _tool_call("get_landscape", {"indication": "als"}, "tc3"),
                ]
            ),
            _ai_final("5 trials found. ALS space is moderately active."),
        ]
    )

    tool_responses_by_turn = [
        {"messages": [_tool_response("tc1", "detect_whitespace", active_whitespace)]},
        {
            "messages": [
                _tool_response("tc2", "search_trials", TRIALS_DATA),
                _tool_response("tc3", "get_landscape", LANDSCAPE_DATA),
            ]
        },
    ]
    mock_tool_node_instance = AsyncMock()
    mock_tool_node_instance.ainvoke = AsyncMock(side_effect=tool_responses_by_turn)

    from unittest.mock import patch

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_agent.ToolNode",
        return_value=mock_tool_node_instance,
    ):
        result = await _run_graph(llm, "riluzole", "als")

    output = result["final_output"]

    # Whitespace
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is False
    assert output.whitespace.exact_match_count == 5

    # Trials
    assert len(output.trials) == 1
    trial = output.trials[0]
    assert trial.nct_id == "NCT02970942"
    assert trial.title == "Riluzole ALS Trial"
    assert trial.phase == "Phase 2"
    assert trial.overall_status == "COMPLETED"
    assert trial.sponsor == "Sponsor Inc"
    assert trial.enrollment == 100
    assert trial.indications == ["ALS"]
    assert len(trial.interventions) == 1
    assert trial.interventions[0].intervention_name == "Riluzole"
    assert trial.references == ["12345678"]

    # Landscape
    assert output.landscape is not None
    assert output.landscape.total_trial_count == 30

    # No terminated
    assert output.terminated == []


# ------------------------------------------------------------------
# assemble_node: final_output is populated from state
# ------------------------------------------------------------------


async def test_assemble_node_produces_final_output():
    """assemble_node assembles ClinicalTrialsOutput from accumulated state, extracting summary from last AIMessage."""
    llm = _make_llm(
        [
            _ai_with_tool_calls(
                [
                    _tool_call(
                        "detect_whitespace",
                        {"drug": "somedrug", "indication": "huntingtons"},
                        "tc1",
                    )
                ]
            ),
            _ai_final(SUMMARY_TEXT),
        ]
    )

    tool_responses_by_turn = [
        {"messages": [_tool_response("tc1", "detect_whitespace", WHITESPACE_DATA)]},
    ]
    mock_tool_node_instance = AsyncMock()
    mock_tool_node_instance.ainvoke = AsyncMock(side_effect=tool_responses_by_turn)

    from unittest.mock import patch

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_agent.ToolNode",
        return_value=mock_tool_node_instance,
    ):
        result = await _run_graph(llm, "somedrug", "huntingtons")

    output = result["final_output"]
    assert output is not None
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is True
    assert output.trials == []
    assert output.terminated == []
    assert output.landscape is None
    assert output.summary == SUMMARY_TEXT


# ------------------------------------------------------------------
# date_before is threaded through to tools
# ------------------------------------------------------------------


async def test_date_before_passed_to_tools():
    """date_before from initial state is used when building tools."""
    cutoff = date(2021, 1, 1)

    llm = _make_llm([_ai_final("Done.")])

    captured_date = []

    from unittest.mock import patch

    original_build = __import__(
        "indication_scout.agents.clinical_trials.clinical_trials_tools",
        fromlist=["build_clinical_trials_tools"],
    ).build_clinical_trials_tools

    def capturing_build(date_before=None, max_search_results=50):
        captured_date.append(date_before)
        return original_build(
            date_before=date_before, max_search_results=max_search_results
        )

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_agent.build_clinical_trials_tools",
        side_effect=capturing_build,
    ):
        await _run_graph(llm, "riluzole", "als", date_before=cutoff)

    assert cutoff in captured_date
