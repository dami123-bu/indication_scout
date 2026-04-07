"""Supervisor agent.

Top-level agent that orchestrates the literature and clinical trials
sub-agents. Given a drug, it surfaces candidate diseases, decides which
to investigate, and delegates to the right sub-agent for each candidate.

The LLM decides:
- Which candidates are worth investigating in depth
- Whether to run literature, clinical trials, or both for each
- When enough evidence has been gathered to stop
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools

SYSTEM_PROMPT = """\
You are a drug repurposing analyst. Given a drug, your job is to find
the most promising new indications and assess them.

You have three tools:

- find_candidates — surfaces candidate diseases for the drug from Open Targets
- analyze_literature — runs a full literature analysis for a drug-disease pair
- analyze_clinical_trials — runs a full clinical trials analysis for a pair

Strategy:
1. Start by surfacing candidates with find_candidates.
2. Pick the most promising candidates to investigate. You don't need to
   analyze every candidate — focus on the ones most likely to yield strong
   repurposing signals based on the disease names and your knowledge of
   the drug's mechanism.
3. For each candidate you investigate, you can run literature analysis,
   clinical trials analysis, or both. Use your judgment:
   - Literature first when you want to assess biological/clinical evidence
   - Clinical trials first when you want to know if the space is already crowded
   - Both for candidates that look genuinely promising
4. You can investigate multiple candidates in parallel by batching tool calls.
5. Stop when you have enough evidence to identify the top candidates, or
   after investigating 3-5 candidates in depth.

GROUNDING RULE: Your final summary must reference ONLY findings
returned by the sub-agent tools in this run. You may use your domain
knowledge to decide which candidates to investigate, but the summary
itself must not introduce trial names, mechanisms, or facts that did
not come back from analyze_literature or analyze_clinical_trials.

End with 4-6 plain sentences summarizing the most promising candidates
and your overall recommendation. No markdown."""


def build_supervisor_agent(llm, svc, db):
    """Return a compiled supervisor agent."""
    tools = build_supervisor_tools(llm=llm, svc=svc, db=db)
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_supervisor_agent(agent, drug_name: str) -> SupervisorOutput:
    """Invoke the supervisor and assemble a SupervisorOutput from the run."""
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Find repurposing opportunities for {drug_name}")
            ]
        }
    )

    candidates: list[str] = []
    # Map disease → CandidateFindings (build incrementally as tool results come in)
    findings_by_disease: dict[str, CandidateFindings] = {}

    # Build a map from tool_call_id → tool_call args so we can recover the
    # disease_name argument that was passed to each analyze_* call
    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[tc["id"]] = tc["args"]

    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue

        if msg.name == "find_candidates":
            candidates = msg.artifact

        elif msg.name == "analyze_literature":
            args = tool_call_args.get(msg.tool_call_id, {})
            disease = args.get("disease_name", "")
            if disease:
                findings = findings_by_disease.setdefault(
                    disease, CandidateFindings(disease=disease)
                )
                findings.literature = msg.artifact

        elif msg.name == "analyze_clinical_trials":
            args = tool_call_args.get(msg.tool_call_id, {})
            disease = args.get("disease_name", "")
            if disease:
                findings = findings_by_disease.setdefault(
                    disease, CandidateFindings(disease=disease)
                )
                findings.clinical_trials = msg.artifact

    # Final AIMessage with no tool calls is the supervisor's narrative summary
    summary = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            summary = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    return SupervisorOutput(
        drug_name=drug_name,
        candidates=candidates,
        findings=list(findings_by_disease.values()),
        summary=summary,
    )
