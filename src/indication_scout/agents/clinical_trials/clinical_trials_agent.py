"""Clinical Trials agent

Uses LangGraph's prebuilt create_react_agent for the agent loop. After
the run, walks the message history to pull typed artifacts off the
ToolMessages and assembles them into a ClinicalTrialsOutput.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)

SYSTEM_PROMPT = """\
You are a clinical trials analyst assessing whether a drug could be
repurposed for a new indication.

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
4. When you have enough information, summarize the findings clearly in 2-3 sentences.No markdown
"""


def build_clinical_trials_agent(llm, date_before=None, max_search_results=None):
    """Return a compiled ReAct agent. No graph wiring required."""
    tools = build_clinical_trials_tools(
        date_before=date_before,
        max_search_results=max_search_results,
    )
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_clinical_trials_agent(
    agent, drug_name: str, disease_name: str
) -> ClinicalTrialsOutput:
    """Invoke the agent and assemble a ClinicalTrialsOutput from the run."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyze {drug_name} in {disease_name}")]}
    )

    artifacts: dict = {
        "whitespace": None,
        "landscape": None,
        "trials": [],
        "terminated": [],
    }

    field_map = {
        "detect_whitespace": "whitespace",
        "search_trials": "trials",
        "get_landscape": "landscape",
        "get_terminated": "terminated",
    }

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in field_map:
            artifacts[field_map[msg.name]] = msg.artifact

    # The final AIMessage with no tool calls is the narrative summary
    summary = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            summary = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return ClinicalTrialsOutput(
        whitespace=artifacts["whitespace"],
        landscape=artifacts["landscape"],
        trials=artifacts["trials"],
        terminated=artifacts["terminated"],
        summary=summary,
    )
