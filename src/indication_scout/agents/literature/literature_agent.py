"""Literature agent — middle-ground version.

Uses LangGraph's prebuilt create_react_agent for the agent loop. No
custom StateGraph, no LiteratureState class, no InjectedState. After
the run, walks the message history to pull typed artifacts off the
ToolMessages and assembles them into a LiteratureOutput.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.literature.literature_tools import build_literature_tools

SYSTEM_PROMPT = """\
You are a biomedical literature researcher investigating whether a drug
could be repurposed for a new indication.

You have five tools:

- build_drug_profile — pharmacological data (synonyms, gene targets, mechanisms)
- expand_search_terms — generates diverse PubMed keyword queries
- fetch_and_cache — runs queries, caches abstracts in the vector store
- semantic_search — re-ranks cached abstracts by relevance
- synthesize — produces a structured evidence summary

Tools that need prior results read them automatically — you do not pass
PMIDs or abstracts as arguments. Batch independent calls when possible.

End with 3-4 plain sentences summarizing the findings. No markdown."""


def build_literature_agent(
    llm,
    svc,
    db,
    date_before=None,
    max_search_results=None,
    num_top_k=5,
):
    """Return a compiled ReAct agent. No graph wiring required."""
    tools = build_literature_tools(
        svc,
        db,
        date_before=date_before,
        max_search_results=max_search_results,
        num_top_k=num_top_k,
    )
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_literature_agent(
    agent, drug_name: str, disease_name: str
) -> LiteratureOutput:
    """Invoke the agent and assemble a LiteratureOutput from the run."""
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Analyze {drug_name} in {disease_name}")
            ]
        }
    )

    # Walk the message history and pull each tool's typed artifact off msg.artifact
    artifacts: dict = {
        "queries": [],
        "pmids": [],
        "abstracts": [],
        "evidence": None,
    }
    # maps tool names → keys in the local artifacts dict , used for mapping to LiteratureOutput
    field_map = {
        "expand_search_terms": "queries",
        "fetch_and_cache": "pmids",
        "semantic_search": "abstracts",
        "synthesize": "evidence",
    }

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in field_map:
            artifacts[field_map[msg.name]] = msg.artifact

    # The final AIMessage with no tool calls is the narrative summary
    summary = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            summary = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    return LiteratureOutput(
        search_results=artifacts["queries"],
        pmids=artifacts["pmids"],
        semantic_search_results=artifacts["abstracts"],
        evidence_summary=artifacts["evidence"],
        summary=summary,
    )
