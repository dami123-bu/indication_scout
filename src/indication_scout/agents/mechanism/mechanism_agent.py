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
You are a drug researcher analysing the molecular targets of a drug to identify
mechanistic repurposing opportunities.

You have three tools:

- get_drug_targets — fetches the molecular targets (gene symbol → Ensembl ID) for the drug
- get_target_associations — fetches the top disease associations for a target, with evidence scores
- get_target_pathways — fetches the Reactome pathways a target participates in

Strategy:
1. Call get_drug_targets first to discover the drug's targets.
2. For each target, call get_target_associations and get_target_pathways.
3. Use the association scores and pathway membership to reason about which
   diseases are mechanistically plausible repurposing candidates.

GROUNDING RULE: Your summary must reference ONLY information returned
by the tools in this run. Do not introduce facts from your training.

End with 3-4 plain sentences summarising the most promising mechanistic
repurposing signals. No markdown."""


def build_mechanism_agent(llm) -> object:
    """Return a compiled ReAct agent."""
    tools = build_mechanism_tools()
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_mechanism_agent(agent, drug_name: str) -> MechanismOutput:
    """Invoke the agent and assemble a MechanismOutput from the run."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyse the targets of {drug_name}")]}
    )

    drug_targets: dict[str, str] = {}
    associations: dict[str, list] = {}
    pathways: dict[str, list] = {}

    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[tc["id"]] = tc["args"]

    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue

        if msg.name == "get_drug_targets":
            drug_targets = msg.artifact or {}

        elif msg.name == "get_target_associations":
            args = tool_call_args.get(msg.tool_call_id, {})
            symbol = args.get("target_symbol", "")
            if symbol:
                associations[symbol] = msg.artifact or []

        elif msg.name == "get_target_pathways":
            args = tool_call_args.get(msg.tool_call_id, {})
            symbol = args.get("target_symbol", "")
            if symbol:
                pathways[symbol] = msg.artifact or []

    summary = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            summary = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    return MechanismOutput(
        drug_targets=drug_targets,
        associations=associations,
        pathways=pathways,
        summary=summary,
    )
