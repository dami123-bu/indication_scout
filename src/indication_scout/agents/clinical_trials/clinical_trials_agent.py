"""Clinical Trials agent

Uses LangGraph's prebuilt create_react_agent for the agent loop. After the run, walks the message
history to pull typed artifacts off the ToolMessages and assembles them into a ClinicalTrialsOutput.
"""

import logging

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You analyze the trial record for a drug × indication pair to assess repurposing potential.

# TOOLS
- check_fda_approval — call this FIRST. Returns is_approved and label_found.
- search_trials — pair query: total + by_status (recruiting/active/withdrawn/unknown) + top 50.
- get_completed — pair query: total + top 50 (read phase off each trial).
- get_terminated — pair query: total + top 50, each with why_stopped text.
- get_landscape — competitive landscape for the indication.
- finalize_analysis — your last action. Plain-text after this is discarded.

# WORKFLOW
Always call ALL of these tools, in parallel: check_fda_approval, search_trials, get_completed,
get_terminated, get_landscape. Then call finalize_analysis with the full summary.

Do NOT skip any tool based on the result of another. In particular, do not skip the trial tools
when check_fda_approval returns is_approved=true or label_found=false — the supervisor still
needs the trial record to reason about subtype/superset relationships and competitive landscape.
Subtype/superset reconciliation is the supervisor's job, not yours.

# WRITING THE SUMMARY
Use plain English. Never use internal field names (by_status, total_count, etc).
Reference only what the tools returned in this run.

Total counts come from total_count fields. Phase counts are derived by looking at the phase
field on each Trial in the returned list — note this is a floor when total_count exceeds the
shown 50 trials. Attribute claims to the registry: "N trials on record as completed Phase 3,"
not "N completed Phase 3 trials" — CT.gov status fields are filings, not ground truth.

When is_approved == false, do NOT say "the drug is not FDA-approved for X." check_fda_approval
matches the candidate string against literal label text and returns false for SUPERSET candidates
(e.g. NAFLD vs approved MASH) or related-but-not-identical names (e.g. cardiovascular disease
vs approved CV risk reduction). The supervisor reconciles these against its briefing; you don't
have it. Say "this exact indication does not appear on the drug's labels in this run" instead.

For Phase 3 trials in completed status with is_approved == false, the trial tools cannot tell
you whether the primary endpoint was met. The trial may have succeeded for a narrower subtype
that is already approved. Phrase outcomes conditionally: "did not lead to approval for this
exact indication," not "the readout was not positive." Surface each trial's title — narrower
subtypes named there (GIST, MASH, DFSP) help the supervisor detect SUPERSET relationships.
For Phase 3 trials completed less than ~2 years ago, the outcome is unknown — say so.

Terminated trials: classify the why_stopped text yourself.
- Safety or efficacy stop on this exact pair → moderate evidence the hypothesis was tested and
  stopped early. Cite the why_stopped text. Do not call the hypothesis "definitively closed."
- Business, enrollment, or other operational reasons → neutral; sponsor decision, not drug
  performance.

UNKNOWN-status trials ran but their outcome is unknowable. Inspect search_trials.trials for
any UNKNOWN entries with Phase 3 (or Phase 2/Phase 3) in the phase field — they are pivotal-
scale activity that won't show up when you count Phase 3 entries in the completed list. Do
not claim "no Phase 3 has been conducted" if such entries exist.

Empty results: search_trials.total_count == 0 means no trial evidence for this pair. Report it
plainly. Still call get_landscape and check_fda_approval.

# DON'T
- Don't describe a completed trial as "sustained clinical interest" — completion is a past event.
- Don't treat absence of terminations as positive evidence. Report it as "no stopping signals on
  record," not "favorable" or "encouraging."
- Don't use outcome-laden words ("validated," "endpoint missed," "failed," "succeeded") unless
  a tool explicitly returned that evidence."""

def build_clinical_trials_agent(llm, date_before=None):
    """Return a compiled ReAct agent. No graph wiring required."""
    tools = build_clinical_trials_tools(
        date_before=date_before,
    )
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_clinical_trials_agent(
    agent, drug_name: str, disease_name: str
) -> ClinicalTrialsOutput:
    """Invoke the agent and assemble a ClinicalTrialsOutput from the run."""
    logger.warning(
        "clinical_trials_agent: starting run for %s × %s", drug_name, disease_name
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyze {drug_name} in {disease_name}")]}
    )

    artifacts: dict = {
        "search": None,
        "completed": None,
        "terminated": None,
        "landscape": None,
        "approval": None,
        "summary": None,
    }

    field_map = {
        "search_trials": "search",
        "get_completed": "completed",
        "get_terminated": "terminated",
        "get_landscape": "landscape",
        "check_fda_approval": "approval",
        "finalize_analysis": "summary",
    }

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in field_map:
            artifacts[field_map[msg.name]] = msg.artifact

    tools_called = [k for k, v in artifacts.items() if v is not None]
    logger.warning(
        "clinical_trials_agent: %s × %s — tools called: %s",
        drug_name,
        disease_name,
        tools_called,
    )

    if artifacts["approval"] is None:
        logger.warning(
            "clinical_trials_agent: %s × %s — check_fda_approval was not called "
            "(prompt requires it as step 1)",
            drug_name,
            disease_name,
        )

    if artifacts["summary"] is None:
        logger.warning(
            "clinical_trials_agent: %s × %s — finalize_analysis was not called; "
            "summary will be empty",
            drug_name,
            disease_name,
        )

    summary = artifacts.get("summary") or ""

    return ClinicalTrialsOutput(
        search=artifacts["search"],
        completed=artifacts["completed"],
        terminated=artifacts["terminated"],
        landscape=artifacts["landscape"],
        approval=artifacts["approval"],
        summary=summary,
    )
