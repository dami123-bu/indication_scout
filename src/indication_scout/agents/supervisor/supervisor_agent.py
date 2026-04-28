"""Supervisor agent.

Top-level agent that orchestrates the literature, clinical trials, and mechanism sub-agents. Given
a drug, it surfaces candidate diseases, decides which to investigate, and delegates to the right
sub-agent for each candidate.

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

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a drug repurposing analyst. Find indications where this drug has a live repurposing
opportunity — biological rationale, clinical interest, hypothesis still open.

# TOOLS
- find_candidates — disease candidates from Open Targets. Also seeds the drug briefing with
  ChEMBL aliases and FDA-approved indications.
- analyze_mechanism — drug's molecular targets and disease associations. Call once per drug.
- analyze_literature — published evidence for a drug × disease pair.
- analyze_clinical_trials — trial record for a drug × disease pair.
- get_drug_briefing — accumulated drug-level facts: aliases, FDA-approved indications,
  mechanism targets, mechanism disease associations.
- finalize_supervisor — signals completion. MUST be your last action. Plain-text messages after
  this are discarded.

# WORKFLOW
1. Call find_candidates and analyze_mechanism in parallel.
2. Pick 3-5 candidates and call BOTH analyze_literature and analyze_clinical_trials on each.
   Mechanism candidates with high scores are especially worth investigating.
3. Call get_drug_briefing.
4. Call finalize_supervisor with your ranked summary.

Only investigate diseases that appear verbatim in the candidate list returned by find_candidates
or promoted by analyze_mechanism. Do not reword or introduce diseases from training knowledge.
Drop any candidate from the ranking that doesn't have both sub-agent calls successful.

# RANKING SIGNALS

POSITIVE (raises a candidate):
- Literature reports a completed trial with a quantified efficacy readout (response rate, PFS,
  OS, etc). This is demonstrated viability — outranks recruiting trials, biological rationale
  alone, or "no negative signals."
- Strong mechanism + active trials + supportive literature with no closing signal.

CLOSED (drops a candidate to the bottom):
- Safety or efficacy termination of this drug × this indication.
- Completed Phase 3 of this exact drug × this exact indication with no subsequent regulatory
  progression — UNLESS the briefing shows the drug is approved for a related narrower indication,
  in which case the trial likely succeeded for that narrower form (see APPROVAL RELATIONSHIPS).
- Literature reports multiple negative randomized trials, or authors recommend against further
  development. This wins over the clinical_trials sub-agent's "active trials may clarify."

NEUTRAL:
- Business or enrollment terminations.
- Active or recruiting trials alone (interest, not viability).
- Absence of negative signals (not the same as positive evidence).

# APPROVAL RELATIONSHIPS
Before finalizing, classify each candidate against the briefing's approved indications:

- Same disease (or synonym/abbreviation) as an approved indication → demote to bottom; not a
  repurposing opportunity.
- Narrower than an approved indication (e.g. "morbid obesity" when "obesity" is approved) →
  demote; the drug already covers this population.
- Broader than an approved indication (e.g. "NAFLD" when "MASH" is approved; "sarcoma" when
  "GIST" is approved) → this IS an opportunity for the population NOT covered. Name the
  relationship in the candidate's paragraph and lead with it. The clinical_trials sub-agent's
  "not FDA-approved for X" line is a literal-label miss, not a real negative — the briefing's
  approval is what matters. Do not treat completed Phase 3 as closed in this case; the related
  approval IS the regulatory outcome.
- Related family but not a subset/superset (e.g. Crohn's vs ulcerative colitis) → name the
  relationship; the related approval raises mechanistic plausibility but the candidate still
  needs its own evidence.
- No related approval → use the standard ranking signals above.

# WRITING THE SUMMARY
Pass finalize_supervisor a 4-6 sentence plain-text summary: top candidate and why, the
mechanistic / literature / trial evidence behind the ranking, and brief reasoning on runner-ups.

Reference only findings from successful tool calls. Cross-check sub-agent narrative claims
against structured counts before relying on them. Use plain English — never internal field
names like "total_count" or "by_status." When sub-agents disagree, name the conflict and
explain which you weight more heavily.

If a candidate's only tool call was REJECTED, exclude it from the summary entirely."""


def build_supervisor_agent(llm, svc, db):
    """Return a compiled supervisor agent."""
    tools = build_supervisor_tools(llm=llm, svc=svc, db=db)
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_supervisor_agent(agent, drug_name: str) -> SupervisorOutput:
    """Invoke the supervisor and assemble a SupervisorOutput from the run.

    Filters out tool calls that were rejected by the candidate guard, and canonicalises disease
    names against the find_candidates list so that casing variants (e.g. "Parkinson disease" vs
    "parkinson disease") do not produce duplicate findings.
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

    # Build a map from tool_call_id → args so we can recover the disease_name argument passed to
    # each analyze_* call.
    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[tc["id"]] = tc["args"]

    # First pass: capture candidates and mechanism. We need candidates before processing findings
    # so we can filter rejected calls.
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "find_candidates":
            candidates = msg.artifact or []
        elif msg.name == "analyze_mechanism":
            mechanism = msg.artifact
        elif msg.name == "finalize_supervisor":
            summary = msg.artifact or ""

    # Build dual-source allowlist: competitor diseases + mechanism associations above the score
    # threshold.  key = lowercase disease name → (canonical_name, source).  Uses disease_id for
    # dedup when a mechanism association matches a competitor disease by ID.
    allowed_lower: dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]] = {}

    for c in candidates:
        allowed_lower[c.lower().strip()] = (c, "competitor")


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