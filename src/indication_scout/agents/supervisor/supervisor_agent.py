"""Supervisor agent.

Top-level agent that orchestrates the literature, clinical trials, and mechanism
sub-agents. Given a drug, it surfaces candidate diseases, decides which
to investigate, and delegates to the right sub-agent for each candidate.

The LLM decides:
- Which candidates are worth investigating in depth
- Whether to run literature, clinical trials, or both for each
- When enough evidence has been gathered to stop
"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools
from indication_scout.constants import MECHANISM_ASSOCIATION_MIN_SCORE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a drug repurposing analyst. Given a drug, your job is to find
the most promising new indications and assess them.

You have five tools:

- find_candidates — surfaces candidate diseases for the drug from Open Targets
- analyze_mechanism — fetches the drug's molecular targets, their disease
  associations, and Reactome pathways; drug-level, call once per drug
- analyze_literature — runs a full literature analysis for a drug-disease pair
- analyze_clinical_trials — runs a full clinical trials analysis for a pair
- finalize_supervisor — signals completion; you MUST call this last

CANDIDATE RULE:
You may ONLY call analyze_literature and analyze_clinical_trials with a
disease_name that appears VERBATIM in the allowed candidate list. Candidates
come from two sources:

1. **Competitor-sourced**: diseases returned by find_candidates (competitor drugs
   are active in these diseases).
2. **Mechanism-sourced**: diseases from analyze_mechanism whose overall
   association score meets the threshold. These are automatically added to the
   allowed list when analyze_mechanism runs — the tool will tell you which
   mechanism diseases were added.

Do not reword, substitute synonyms, expand abbreviations, or introduce
diseases from your training knowledge.

Examples of disallowed substitutions:
- find_candidates returns "fatty liver disease" → do NOT pass "NASH" or "MASH"
- find_candidates returns "cardiovascular disease" → do NOT pass "heart failure"

If a disease is rejected by a tool call, it is NOT in the allowed list.

Strategy:
1. Call find_candidates and analyze_mechanism in parallel as your first step.
2. Use find_candidates to see which diseases competitor drugs are active in.
   Use analyze_mechanism for both mechanistic context AND to surface additional
   candidate diseases based on target-disease associations.
3. Pick 3-5 candidates to investigate in depth with literature and/or trials.
   Mechanism-sourced candidates with high association scores can be especially
   interesting — they represent biologically plausible repurposing hypotheses.
   Use your judgment:
   - Literature first when you want to assess biological/clinical evidence
   - Clinical trials first when you want to know if the space is already crowded
   - Both for candidates that look genuinely promising
4. You can investigate multiple candidates in parallel by batching tool calls.
5. Stop when you have enough evidence to identify the top candidates.

REJECTION HANDLING:
If a tool call returns a message starting with "REJECTED:", that call
produced NO data. The disease in that rejected call MUST NOT appear in
your final summary. Retry with a valid candidate name if needed.

GROUNDING RULE:
Your final summary must reference ONLY findings returned by SUCCESSFUL
(non-rejected) sub-agent tool calls in this run. Before calling
finalize_supervisor, review your tool history — if a disease's only
tool call was REJECTED, do not include that disease in your summary.

IMPORTANT: finalize_supervisor MUST be the final tool call. Pass it your
4-6 sentence plain-text summary of the most promising candidates,
referencing relevant mechanistic context from analyze_mechanism where it
supports or contextualises the findings. Do NOT emit a plain
text message after calling finalize_supervisor."""


def build_supervisor_agent(llm, svc, db):
    """Return a compiled supervisor agent."""
    tools = build_supervisor_tools(llm=llm, svc=svc, db=db)
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_supervisor_agent(agent, drug_name: str) -> SupervisorOutput:
    """Invoke the supervisor and assemble a SupervisorOutput from the run.

    Filters out tool calls that were rejected by the candidate guard, and
    canonicalises disease names against the find_candidates list so that
    casing variants (e.g. "Parkinson disease" vs "parkinson disease") do
    not produce duplicate findings.
    """
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Find repurposing opportunities for {drug_name}")
            ]
        }
    )

    candidates: list[str] = []
    mechanism: MechanismOutput | None = None
    summary: str = ""
    findings_by_disease: dict[str, CandidateFindings] = {}

    # Build a map from tool_call_id → args so we can recover the disease_name
    # argument passed to each analyze_* call.
    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[tc["id"]] = tc["args"]

    # First pass: capture candidates and mechanism. We need candidates before
    # processing findings so we can filter rejected calls.
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "find_candidates":
            candidates = msg.artifact or []
        elif msg.name == "analyze_mechanism":
            mechanism = msg.artifact
        elif msg.name == "finalize_supervisor":
            summary = msg.artifact or ""

    # Build dual-source allowlist: competitor diseases + mechanism associations
    # above the score threshold.  key = lowercase disease name →
    # (canonical_name, source).  Uses disease_id for dedup when a mechanism
    # association matches a competitor disease by ID.
    allowed_lower: dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]] = {}

    for c in candidates:
        allowed_lower[c.lower().strip()] = (c, "competitor")

    # Build a disease_id → lowercase-key index from competitor diseases so we
    # can dedup mechanism associations by ID rather than string matching alone.
    competitor_id_to_key: dict[str, str] = {}
    if mechanism:
        # We don't have disease_ids on the competitor list directly, but
        # mechanism associations carry disease_id. If a mechanism association's
        # disease_name already matches a competitor key, record its ID.
        for assoc_list in mechanism.associations.values():
            for assoc in assoc_list:
                key = assoc.disease_name.lower().strip()
                if key in allowed_lower and assoc.disease_id:
                    competitor_id_to_key[assoc.disease_id] = key

        # Now add qualifying mechanism associations.
        for assoc_list in mechanism.associations.values():
            for assoc in assoc_list:
                if (assoc.overall_score or 0) < MECHANISM_ASSOCIATION_MIN_SCORE:
                    continue
                key = assoc.disease_name.lower().strip()

                # Check by disease_id first (handles name mismatches for same disease)
                existing_key = competitor_id_to_key.get(assoc.disease_id)
                if existing_key:
                    name, source = allowed_lower[existing_key]
                    if source == "competitor":
                        allowed_lower[existing_key] = (name, "both")
                elif key in allowed_lower:
                    name, source = allowed_lower[key]
                    if source == "competitor":
                        allowed_lower[key] = (name, "both")
                else:
                    allowed_lower[key] = (assoc.disease_name, "mechanism")

    def _canonical(disease_raw: str) -> tuple[str, Literal["competitor", "mechanism", "both"]] | None:
        """Return (canonical_name, source) for disease_raw, or None if not allowed."""
        key = disease_raw.lower().strip()
        return allowed_lower.get(key)

    # Second pass: assemble findings, skipping rejected calls and canonicalising keys.
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name not in {"analyze_literature", "analyze_clinical_trials"}:
            continue

        args = tool_call_args.get(msg.tool_call_id, {})
        disease_raw = args.get("disease_name", "")
        match = _canonical(disease_raw)

        if match is None:
            logger.warning(
                "Skipping rejected %s call for disease=%r (not in allowlist)",
                msg.name,
                disease_raw,
            )
            continue

        canonical, source = match
        findings = findings_by_disease.setdefault(
            canonical, CandidateFindings(disease=canonical, source=source)
        )

        if msg.name == "analyze_literature":
            findings.literature = msg.artifact
        else:  # analyze_clinical_trials
            findings.clinical_trials = msg.artifact

    return SupervisorOutput(
        drug_name=drug_name,
        candidates=candidates,
        mechanism=mechanism,
        findings=list(findings_by_disease.values()),
        summary=summary,
    )