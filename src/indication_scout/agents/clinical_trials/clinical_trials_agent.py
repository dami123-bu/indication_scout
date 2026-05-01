"""Clinical Trials agent

Uses LangGraph's prebuilt create_react_agent for the agent loop. After the run, walks the message
history to pull typed artifacts off the ToolMessages and assembles them into a ClinicalTrialsOutput.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
1. Call check_fda_approval first.
2. If is_approved == true → call finalize_analysis immediately with one sentence saying the drug
   is FDA-approved for the indication. Do not call any other tool.
3. If label_found == false → call finalize_analysis immediately with one sentence saying no FDA
   label was found and approval status is unknown. Do not call any other tool.
4. Otherwise → call search_trials, get_completed, get_terminated, get_landscape (in parallel),
   then finalize_analysis with the full summary.

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

    # Diagnostic: count turns + characterize the final message so we can tell
    # whether the loop hit the recursion limit, the LLM stopped emitting tool
    # calls without calling finalize_analysis, or finalize_analysis was called
    # with an empty/whitespace string.
    msgs = result["messages"]
    n_messages = len(msgs)
    n_tool_messages = sum(1 for m in msgs if isinstance(m, ToolMessage))
    n_ai_messages = sum(1 for m in msgs if isinstance(m, AIMessage))
    last_msg = msgs[-1] if msgs else None
    last_msg_type = type(last_msg).__name__ if last_msg is not None else "None"
    last_ai_had_tool_calls = (
        bool(getattr(last_msg, "tool_calls", None))
        if isinstance(last_msg, AIMessage)
        else None
    )
    last_ai_text_len = (
        len(last_msg.content) if isinstance(last_msg, AIMessage) and isinstance(last_msg.content, str) else None
    )
    logger.warning(
        "clinical_trials_agent: %s × %s — turn shape: total_msgs=%d, "
        "tool_msgs=%d, ai_msgs=%d, last=%s, last_ai_tool_calls=%s, "
        "last_ai_text_len=%s",
        drug_name,
        disease_name,
        n_messages,
        n_tool_messages,
        n_ai_messages,
        last_msg_type,
        last_ai_had_tool_calls,
        last_ai_text_len,
    )

    if artifacts["approval"] is None:
        logger.warning(
            "clinical_trials_agent: %s × %s — check_fda_approval was not called "
            "(prompt requires it as step 1)",
            drug_name,
            disease_name,
        )

    if artifacts["summary"] is None:
        # Distinguish three failure modes:
        #   A. finalize_analysis ToolMessage entirely absent (LLM never called
        #      it — likely recursion-limit hit or LLM emitted final text
        #      instead of a tool call).
        #   B. finalize_analysis called but artifact is None (shouldn't
        #      happen given the tool returns the summary string verbatim, but
        #      pin it just in case).
        finalize_msgs = [
            m for m in msgs if isinstance(m, ToolMessage) and m.name == "finalize_analysis"
        ]
        if not finalize_msgs:
            logger.warning(
                "clinical_trials_agent: %s × %s — finalize_analysis was not called; "
                "summary will be empty (last_msg=%s, last_ai_had_tool_calls=%s, "
                "last_ai_text_len=%s)",
                drug_name,
                disease_name,
                last_msg_type,
                last_ai_had_tool_calls,
                last_ai_text_len,
            )
            # If the final AIMessage has substantial text content but no tool
            # calls, the LLM tried to write the summary inline instead of
            # routing it through finalize_analysis. Capture an excerpt.
            if (
                isinstance(last_msg, AIMessage)
                and isinstance(last_msg.content, str)
                and last_msg.content.strip()
            ):
                preview = last_msg.content.strip()[:300]
                logger.warning(
                    "clinical_trials_agent: %s × %s — final AIMessage carried "
                    "free text instead of a finalize_analysis tool call: %r",
                    drug_name,
                    disease_name,
                    preview,
                )
        else:
            logger.warning(
                "clinical_trials_agent: %s × %s — finalize_analysis ToolMessage "
                "found but artifact is None (count=%d)",
                drug_name,
                disease_name,
                len(finalize_msgs),
            )

    # Detect the case where finalize_analysis WAS called but with empty /
    # whitespace-only input. The artifact is the summary string; this catches
    # `finalize_analysis("")`.
    if artifacts["summary"] is not None and not str(artifacts["summary"]).strip():
        logger.warning(
            "clinical_trials_agent: %s × %s — finalize_analysis was called with "
            "empty/whitespace summary (raw=%r)",
            drug_name,
            disease_name,
            artifacts["summary"],
        )

    # Always log the summary artifact's length + preview so we can correlate
    # short/odd summaries with downstream rendering issues. This is the actual
    # text that goes into the report, distinct from last_ai_text_len above.
    if artifacts["summary"] is not None:
        raw_summary = str(artifacts["summary"])
        preview = raw_summary[:300].replace("\n", "\\n")
        logger.warning(
            "clinical_trials_agent: %s × %s — summary len=%d, preview=%r",
            drug_name,
            disease_name,
            len(raw_summary),
            preview,
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
