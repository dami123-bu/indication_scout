"""Clinical Trials agent

Uses LangGraph's prebuilt create_react_agent for the agent loop. After
the run, walks the message history to pull typed artifacts off the
ToolMessages and assembles them into a ClinicalTrialsOutput.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)
from indication_scout.models.model_clinical_trials import TrialOutcomes

SYSTEM_PROMPT = """\
You are a clinical trials analyst assessing whether a drug could be
repurposed for a new indication.

You have five tools:

- detect_whitespace — checks if any trials exist for this drug-indication pair
- search_trials — fetches details on trials matching the drug and indication
- get_landscape — competitive landscape: total trials, top sponsors, phase distribution
- get_terminated — trial-outcome evidence split into four scopes: drug_wide
  (this drug in ANY indication, safety/efficacy only), indication_wide
  (this indication with ANY drug), pair_specific (this drug in this
  indication, TERMINATED — all stop_categories), and pair_completed
  (this drug in this indication, COMPLETED — trials that ran to protocol
  end). Report scopes separately. A pair_specific safety/efficacy
  termination OR a completed pair Phase 3 without follow-on development
  both indicate the exact hypothesis has been tested and did not
  succeed.
- finalize_analysis — signals completion; you MUST call this last

Decide which tools to call based on what you learn. Typically start with
detect_whitespace. ALWAYS call get_terminated alongside search_trials and
get_landscape — terminated trials are evidence whether or not active trials
exist. Do not skip get_terminated just because other trials were found.

Batch independent tool calls into a single response when possible.

IMPORTANT: finalize_analysis MUST be the final tool call. Pass it your
2-3 sentence plain-text summary of the findings. Do NOT
emit a plain text message after calling finalize_analysis.

COMPETITIVE INTENSITY RULES — apply these when summarising the landscape:
- Any Phase 3 trial with >500 enrolled participants from a named pharma/biotech
  sponsor signals a HIGH-INVESTMENT, CROWDED space — do not describe it as early
  or uncrowded regardless of total trial count.
- A completed Phase 3 trial means the space has reached late-stage validation;
  flag this explicitly.
- "Few trials" (e.g. 6) does not mean "uncrowded" if those trials are large Phase 3
  programmes. Distinguish trial count from investment scale.
- When reporting competitive intensity, cite the highest-phase trials, their
  enrollment, and their status — not just the total count.
- When reporting Phase 3 trial counts, check enrollment and primary outcome
  measure for each trial. A Phase 3 with enrollment under ~50 or a primary
  outcome that is a biomarker, gene expression, PK parameter, or imaging/CSF
  measurement is a mechanistic substudy, not a pivotal efficacy trial —
  report it separately or exclude it from the pivotal count.

TRIAL-OUTCOME RULES — apply these when weighing trial evidence:
- pair_specific terminations with safety or efficacy stop_category mean
  THIS drug has already been tested in THIS indication and stopped
  early. Treat this as a closed hypothesis.
- pair_completed Phase 3 trials mean THIS drug has run a full protocol
  in THIS indication. A completed Phase 3 without subsequent regulatory
  progression (no approval, no follow-on Phase 3) is strong evidence
  the primary endpoint was missed. Treat as closed unless there is
  direct evidence the readout was positive. Do not frame pair_completed
  Phase 3 trials as "sustained clinical interest" — a completed trial
  is a settled question, not an open one.
- drug_wide safety/efficacy terminations reflect the drug's broader
  failure history and should temper optimism across all candidates.
- indication_wide terminations reflect how hard the disease area is
  historically; weigh them as context, not as closure of this drug.
- Business or enrollment terminations (in any scope) are neutral —
  they reflect sponsor decisions, not drug performance.

GROUNDING RULE: Your summary must reference ONLY information returned
by the tools in this run. Do not introduce trial names, drug histories,
or facts from your training that were not returned by the tools."""

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
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyze {drug_name} in {disease_name}")]}
    )

    artifacts: dict = {
        "whitespace": None,
        "landscape": None,
        "trials": [],
        "terminated": TrialOutcomes(),
        "summary": None,
    }

    field_map = {
        "detect_whitespace": "whitespace",
        "search_trials": "trials",
        "get_landscape": "landscape",
        "get_terminated": "terminated",
        "finalize_analysis": "summary",
    }

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in field_map:
            artifacts[field_map[msg.name]] = msg.artifact

    summary = artifacts.get("summary") or ""

    return ClinicalTrialsOutput(
        whitespace=artifacts["whitespace"],
        landscape=artifacts["landscape"],
        trials=artifacts["trials"],
        terminated=artifacts["terminated"],
        summary=summary,
    )
