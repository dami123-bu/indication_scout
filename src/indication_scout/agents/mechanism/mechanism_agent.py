"""Mechanism agent.

Uses LangGraph's prebuilt create_react_agent for the agent loop. After
the run, walks the message history to pull typed artifacts off the
ToolMessages and assembles them into a MechanismOutput.
"""
import logging

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.mechanism.mechanism_tools import build_mechanism_tools
from indication_scout.models.model_open_targets import MechanismOfAction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = ""


def build_mechanism_agent(llm) -> object:
    """Return a compiled ReAct agent."""
    tools = build_mechanism_tools()
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_mechanism_agent(agent, drug_name: str) -> MechanismOutput:
    """Invoke the agent and assemble a MechanismOutput from the run."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyse the targets of {drug_name}")]}
    )

    mechanisms_of_action: list[MechanismOfAction] = []
    associations: dict[str, list] = {}
    summary: str = ""

    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue

        if msg.name == "get_drug":
            mechanisms_of_action = msg.artifact or []

        elif msg.name == "get_target_associations":
            associations.update(msg.artifact or {})

        elif msg.name == "finalize_analysis":
            summary = msg.artifact or ""

    # Derive drug_targets from mechanisms_of_action (already fetched by get_drug)
    drug_targets: dict[str, str] = {
        symbol: target_id
        for moa in mechanisms_of_action
        for symbol, target_id in zip(moa.target_symbols, moa.target_ids)
    }

    all_mech_diseases = {
        a.disease_name
        for assoc_list in associations.values()
        for a in assoc_list
    }
    logger.warning("[MECH] surfaced %d diseases: %s",
                   len(all_mech_diseases), sorted(all_mech_diseases))

    return MechanismOutput(
        drug_targets=drug_targets,
        mechanisms_of_action=mechanisms_of_action,
        summary=summary,
    )
