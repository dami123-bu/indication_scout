import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.literature.literature_state import LiteratureState
from indication_scout.agents.literature.literature_tools import build_literature_tools

logger = logging.getLogger(__name__)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


def build_literature_graph(
    llm, svc, db, drug_profile, date_before=None, max_search_results=None
):

    tools = build_literature_tools(
        svc,
        db,
        drug_profile,
        date_before=date_before,
        max_search_results=max_search_results,
    )

    model_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # System prompt template
    SYSTEM_PROMPT = """You are an expert biomedical literature researcher.

Your goal is to identify published evidence for a drug in a specific disease/indication.

Current Context:
- Drug: {drug_name}
- Disease/Indication: {disease_name}
- Date Cutoff: {date_before}

Instructions:
1. Always use the current drug and disease shown above when calling tools.
2. Call `expand_search_terms` to generate diverse PubMed keyword queries for this drug-disease pair.
3. Call `fetch_and_cache` passing the search terms returned by `expand_search_terms`.
4. When done, summarize the findings briefly in 1-2 sentences."""

    # ====================== NODES ======================

    async def agent_node(state: LiteratureState, config=None):
        """LLM decides what to do with a clear system prompt."""

        # Format the system prompt with current context
        system_content = SYSTEM_PROMPT.format(
            drug_name=state.drug_name or "Not specified",
            disease_name=state.disease_name or "Not specified",
            date_before=state.date_before or "None",
        )

        system_prompt = SystemMessage(content=system_content)

        # Combine system prompt + conversation history
        messages_to_send = [system_prompt] + list(state.messages)

        response = await model_with_tools.ainvoke(messages_to_send, config=config)

        return {"messages": [response]}

    async def tools_node(state: LiteratureState):

        tool_results = await tool_node.ainvoke(state)
        updates: dict[str, Any] = {"messages": tool_results["messages"]}

        # Dispatch table: maps each tool name to the LiteratureState field that should
        # receive its output, and the expected type. Used to generically process
        # ToolMessages without a chain of if/elif checks per tool.
        tool_handlers = {"expand_search_terms": ("search_results", list), "fetch_and_cache": ("pmids", lambda d: d["pmids"])}

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
                    logger.debug("Parsed %s into state.%s", tool_name, field_name)
            except Exception as e:
                logger.error("Failed to parse %s: %s", tool_name, e)

        return updates

    def assemble_node(state: LiteratureState):
        """Final step: combine everything into one output object.

        The message history looks like:
            HumanMessage        ← initial user prompt
            AIMessage           ← LLM decides to call tool(s)
            ToolMessage(s)      ← tool results
            AIMessage           ← LLM decides to call more tools, or produces final summary
            ToolMessage(s)      ← tool results (if another round)
            AIMessage           ← final plain-text summary (no tool calls)

        We want the final AIMessage that comes *after* the last ToolMessage —
        that is the LLM's narrative summary, not an intermediate tool-calling step.
        """
        messages = list(state.messages)

        # Find the index of the last ToolMessage so we can ignore AI messages
        # that appear before or at that point (those are tool-calling steps, not summaries).
        last_tool_index = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage):
                last_tool_index = i

        # Walk backwards through messages looking for the first AIMessage that
        # appears after the last tool result and contains non-empty text.
        summary = ""
        for i, msg in reversed(list(enumerate(messages))):
            if not isinstance(msg, AIMessage) or i <= last_tool_index:
                continue
            # Claude returns content as a plain string when there are no tool calls.
            if isinstance(msg.content, str) and msg.content.strip():
                summary = msg.content.strip()
                break
            # Claude returns content as a list of blocks when mixing text + tool_use.
            # Extract only the text blocks and join them.
            if isinstance(msg.content, list):
                text_parts = [
                    block.get("text", "")
                    for block in msg.content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if text_parts:
                    summary = "\n".join(text_parts).strip()
                    break

        final_output = LiteratureOutput(
            search_results=state.search_results,
            pmids=state.pmids,
            summary=summary,
        )

        return {
            "final_output": final_output,
            "messages": [AIMessage(content="Literature analysis completed.")],
        }

    # ====================== BUILD GRAPH ======================

    workflow = StateGraph(LiteratureState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("assemble", assemble_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", "__end__": "assemble"}
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("assemble", END)

    return workflow.compile()
