"""Literature agent — agentic LangGraph version.

The LLM decides which tools to call and in what order. Tools read
inter-step dependencies (pmids, abstracts) from a shared closure store
that tools_node updates after each round.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.literature.literature_state import LiteratureState
from indication_scout.agents.literature.literature_tools import build_literature_tools

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a biomedical literature researcher investigating whether a drug
could be repurposed for a new indication.

Drug: {drug_name}
Disease/Indication: {disease_name}
Date Cutoff: {date_before}

You have five tools:

- build_drug_profile — fetches pharmacological data (synonyms, gene targets,
  mechanisms) for the drug. Useful for generating better search queries.
- expand_search_terms — generates diverse PubMed keyword queries for the
  drug-disease pair. Uses the drug profile if available.
- fetch_and_cache — runs PubMed queries and caches abstracts in the vector
  store. Reads queries from prior results automatically.
- semantic_search — re-ranks cached abstracts by relevance to the drug-disease
  pair. Reads PMIDs from prior results automatically.
- synthesize — produces a structured evidence summary from the top abstracts.
  Reads abstracts from prior results automatically. Call even when evidence
  is weak — negative findings are valuable.

Tools that need prior results read them automatically — you do not need
to pass PMIDs or abstracts as arguments.

Batch independent tool calls into a single response when possible.

IMPORTANT: When done, your final message must be ONLY 3-4 plain text
sentences summarizing the findings. No markdown, no tables, no headers,
no bullet points. The structured data is already captured by the tools."""


def build_literature_graph(
    llm,
    svc,
    db,
    date_before=None,
    max_search_results=None,
    num_top_k=5,
):
    tools, store = build_literature_tools(
        svc,
        db,
        date_before=date_before,
        max_search_results=max_search_results,
        num_top_k=num_top_k,
    )

    model_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # Maps tool names to both state fields and store keys
    artifact_fields = {
        "build_drug_profile": "drug_profile",
        "expand_search_terms": "expanded_search_results",
        "fetch_and_cache": "pmids",
        "semantic_search": "semantic_search_results",
        "synthesize": "evidence_summary",
    }

    # ── nodes ──────────────────────────────────────────────────

    async def agent_node(state: LiteratureState, config=None):
        system_content = SYSTEM_PROMPT.format(
            drug_name=state.drug_name or "Not specified",
            disease_name=state.disease_name or "Not specified",
            date_before=state.date_before or "None",
        )
        messages_to_send = [SystemMessage(content=system_content)] + list(
            state.messages
        )
        response = await model_with_tools.ainvoke(messages_to_send, config=config)
        return {"messages": [response]}

    async def tools_node(state: LiteratureState):
        tool_results = await tool_node.ainvoke(state)
        updates: dict[str, Any] = {"messages": tool_results["messages"]}

        for msg in tool_results.get("messages", []):
            if isinstance(msg, ToolMessage) and msg.name in artifact_fields:
                field = artifact_fields[msg.name]
                updates[field] = msg.artifact
                # Keep the shared store in sync so downstream tools can read it
                store[field] = msg.artifact
                logger.debug(
                    "Stored %s artifact in state.%s and store",
                    msg.name,
                    field,
                )

        return updates

    def assemble_node(state: LiteratureState):
        messages = list(state.messages)

        last_tool_idx = max(
            (i for i, m in enumerate(messages) if isinstance(m, ToolMessage)),
            default=-1,
        )

        summary = ""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, AIMessage) or i <= last_tool_idx:
                continue
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

        return {
            "final_output": LiteratureOutput(
                search_results=state.expanded_search_results,
                pmids=state.pmids,
                semantic_search_results=state.semantic_search_results,
                evidence_summary=state.evidence_summary,
                summary=summary,
            ),
            "messages": [AIMessage(content="Literature analysis completed.")],
        }

    # ── graph ──────────────────────────────────────────────────

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
