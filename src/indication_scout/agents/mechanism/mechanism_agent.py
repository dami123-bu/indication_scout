"""Mechanism agent.

Uses LangGraph's prebuilt create_react_agent for the agent loop. After
the run, walks the message history to pull typed artifacts off the
ToolMessages and assembles them into a MechanismOutput.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.mechanism.mechanism_tools import build_mechanism_tools


SYSTEM_PROMPT = """\
You are a drug researcher surfacing repurposing opportunities for a drug.

You have three tools:

- get_drug_competitors — finds candidate diseases based on competitor drugs
  that share the same molecular targets
- build_drug_profile — pharmacological profile (synonyms, gene targets, mechanisms)
- expand_search_terms — generates PubMed search queries for a drug-disease pair

Strategy: start by getting the drug profile and the candidate diseases.
Then generate search terms for candidates that look most promising as
repurposing opportunities. You don't need to generate terms for every
candidate — focus on the ones where the drug's mechanism plausibly
connects to the disease.

GROUNDING RULE: Your summary must reference ONLY information returned
by the tools in this run. Do not introduce facts from your training.

End with 3-4 plain sentences summarizing what you found. No markdown."""


def build_mechanism_agent(llm, svc, date_before=None):
    """Return a compiled ReAct agent. No graph wiring required."""
    tools = build_mechanism_tools(svc)
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_mechanism_agent(agent, drug_name: str) -> MechanismOutput:
    """Invoke the agent and assemble a MechanismOutput from the run."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyze {drug_name}")]}
    )

    competitors_raw = None
    drug_profile = None
    search_queries: dict[str, list[str]] = {}

    # Build a map from tool_call_id → tool_call args so we can recover
    # the disease argument that was passed to each expand_search_terms call
    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[tc["id"]] = tc["args"]

    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "get_drug_competitors":
            competitors_raw = msg.artifact
        elif msg.name == "build_drug_profile":
            drug_profile = msg.artifact
        elif msg.name == "expand_search_terms":
            args = tool_call_args.get(msg.tool_call_id, {})
            disease = args.get("disease", "")
            if disease:
                search_queries[disease] = msg.artifact

    # The final AIMessage with no tool calls is the narrative summary
    summary = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            summary = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    return MechanismOutput(
        competitors={
            d: list(drugs)
            for d, drugs in (
                competitors_raw["diseases"] if competitors_raw else {}
            ).items()
        },
        drug_indications=competitors_raw["drug_indications"] if competitors_raw else [],
        drug_profile=drug_profile,
        search_queries=search_queries,
        summary=summary,
    )