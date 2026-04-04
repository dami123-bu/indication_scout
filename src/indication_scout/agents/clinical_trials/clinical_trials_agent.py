from indication_scout.agents.clinical_trials.clinical_trials_output import ClinicalTrialsOutput
from indication_scout.agents.clinical_trials.clinical_trials_state import ClinicalTrialsState
from indication_scout.agents.clinical_trials.clinical_trials_tools import build_clinical_trials_tools

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

logger = logging.getLogger(__name__)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from indication_scout.models.model_clinical_trials import (
    WhitespaceResult,
    Trial,
    IndicationLandscape,
    TerminatedTrial
)


def build_clinical_trials_graph(llm):
    """
    ClinicalTrialsAgent using LangGraph with system prompt.
    """

    # System prompt template
    SYSTEM_PROMPT = """You are an expert Clinical Trials Research Assistant.

Your goal is to gather comprehensive information about a drug in a specific disease/indication.

Current Context:
- Drug: {drug_name}
- Disease/Indication: {disease_name}
- Date Cutoff: {date_before}

Instructions:
1. Always use the current drug and disease shown above when calling tools.
2. Start with `detect_whitespace` to understand if there's opportunity in this drug-indication pair.
3. Based on the result, call remaining tools in a single batch where possible:
   - If trials exist: call `search_trials` and `get_landscape` together in one response.
   - If whitespace: call `get_terminated` and `get_landscape` together in one response.
   - If `get_terminated` finds safety/efficacy failures, you may skip `get_landscape`.
4. When you have enough information, summarize the findings clearly in 2-3 sentences.

Always batch independent tool calls into a single response to minimise round-trips."""

    # Helper to create tools
    def get_tools(date_before=None):
        return build_clinical_trials_tools(date_before=date_before)

    # ====================== NODES ======================

    async def agent_node(state: ClinicalTrialsState, config=None):
        """LLM decides what to do with a clear system prompt."""
        tools = get_tools(state.date_before)

        # Format the system prompt with current context
        system_content = SYSTEM_PROMPT.format(
            drug_name=state.drug_name or "Not specified",
            disease_name=state.disease_name or "Not specified",
            date_before=state.date_before or "None"
        )

        system_prompt = SystemMessage(content=system_content)

        # Combine system prompt + conversation history
        messages_to_send = [system_prompt] + list(state.messages)

        model_with_tools = llm.bind_tools(tools)
        response = await model_with_tools.ainvoke(messages_to_send, config=config)

        return {"messages": [response]}

    async def tools_node(state: ClinicalTrialsState):
        tools = get_tools(state.date_before)
        tool_node = ToolNode(tools)

        tool_results = await tool_node.ainvoke(state)
        updates: dict[str, Any] = {"messages": tool_results["messages"]}

        tool_handlers = {
            "detect_whitespace": ("whitespace", WhitespaceResult.model_validate),
            "search_trials": ("trials",
                              lambda data: [Trial.model_validate(t) for t in data] if isinstance(data, list) else []),
            "get_landscape": ("landscape", IndicationLandscape.model_validate),
            "get_terminated": ("terminated",
                               lambda data: [TerminatedTrial.model_validate(t) for t in data] if isinstance(data,
                                                                                                            list) else []),
        }

        for msg in tool_results.get("messages", []):
            if not isinstance(msg, ToolMessage):
                continue

            tool_name = msg.name
            content = msg.content

            try:
                data = json.loads(content) if isinstance(content, str) else content

                if tool_name in tool_handlers:
                    field_name, parser = tool_handlers[tool_name]
                    updates[field_name] = parser(data)
                    print(f"✓ Parsed {tool_name} into state.{field_name}")  # helpful debug
            except Exception as e:
                print(f"[ERROR] Failed to parse {tool_name}: {e}")

        return updates

    def assemble_node(state: ClinicalTrialsState):
        """Final step: combine everything into one output object."""
        # Extract summary from the last AI message that has non-empty text content
        summary = ""
        for msg in reversed(list(state.messages)):
            if isinstance(msg, AIMessage):
                if isinstance(msg.content, str) and msg.content.strip():
                    summary = msg.content.strip()
                    break
                if isinstance(msg.content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in msg.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    if text_parts:
                        summary = "\n".join(text_parts).strip()
                        break

        final_output = ClinicalTrialsOutput(
            trials=state.trials,
            whitespace=state.whitespace,
            landscape=state.landscape,
            terminated=state.terminated,
            summary=summary,
        )

        return {
            "final_output": final_output,
            "messages": [AIMessage(content="Clinical trials analysis completed.")]
        }

    # ====================== BUILD GRAPH ======================

    workflow = StateGraph(ClinicalTrialsState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("assemble", assemble_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": "assemble"}
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("assemble", END)

    return workflow.compile()