"""Mechanism agent.

Uses LangGraph's prebuilt create_react_agent for the agent loop. After
the run, walks the message history to pull typed artifacts off the
ToolMessages and assembles them into a MechanismOutput.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.mechanism.mechanism_tools import build_mechanism_tools


SYSTEM_PROMPT = """\
You are a drug researcher analysing the molecular targets of a drug to identify
mechanistic repurposing opportunities.

You have four tools:

- get_drug_targets — fetches the molecular targets (gene symbol → Ensembl ID) for the drug
- get_target_associations — fetches the top disease associations for a target, with evidence scores
- get_target_pathways — fetches the Reactome pathways a target participates in
- finalize_analysis — signals completion; you MUST call this last

Strategy:
1. Call get_drug_targets first to discover the drug's targets.
2. For each target, call get_target_associations and get_target_pathways.
3. Use the association scores and pathway membership to reason about which
   diseases are mechanistically plausible repurposing candidates.

IMPORTANT: finalize_analysis MUST be the final tool call. Pass it your
3-4 sentence plain-text summary of the mechanistic findings (no markdown).
Do NOT emit a plain text message after calling finalize_analysis.

GROUNDING RULE: Your summary must reference ONLY information returned
by the tools in this run. Do not introduce facts from your training."""


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
    summary: str = ""

    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue

        if msg.name == "get_drug_targets":
            drug_targets = msg.artifact or {}

        elif msg.name == "get_target_associations":
            associations.update(msg.artifact or {})

        elif msg.name == "get_target_pathways":
            pathways.update(msg.artifact or {})

        elif msg.name == "finalize_analysis":
            summary = msg.artifact or ""

    return MechanismOutput(
        drug_targets=drug_targets,
        associations=associations,
        pathways=pathways,
        summary=summary,
    )
