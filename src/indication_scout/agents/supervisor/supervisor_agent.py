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
1. Call find_candidates AND analyze_mechanism in parallel. Both are REQUIRED — analyze_mechanism
   is not optional even if find_candidates returns plenty of candidates. The investigation
   tools (analyze_literature, analyze_clinical_trials) will block until both have completed.
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
The summary is a STRUCTURED FACT LIST, not a narrative. No prose paragraphs, no interpretive
adjectives ("promising", "biologically grounded", "active interest", "well-tolerated"), no
mechanism rationale weaving, no comparison framing across candidates. Just verifiable facts
pulled directly from the tool artifacts you received this run.

FORMAT: a ranked list of investigated candidates. For each candidate, one line with these
fields, in this order, separated by semicolons:

  <rank>. <disease> — literature: <strength>, <N> PMIDs; trials: <N> total, <N> completed,
  <N> terminated[, <K> safety/efficacy]; FDA approval: <yes|no>.

Rules:
- Order candidates from strongest to weakest evidence. Ties broken by trial count.
- "literature: <strength>" must be the verbatim strength label from analyze_literature
  (none / weak / moderate / strong). "<N> PMIDs" is len(output.pmids) from that artifact.
- Trial counts come from analyze_clinical_trials structured counts. If a scope is missing,
  write 0 — never estimate.
- Include "<K> safety/efficacy" only when terminated > 0 and at least one termination
  classified as safety or efficacy (per the structured count you were shown).
- "FDA approval: yes" only if the clinical_trials artifact's approval.is_approved is true.

After the ranked list, add ONE optional final line, prefixed "Closed signals:" naming any
candidate(s) with a safety or efficacy termination, or completed Phase 3 without approval. No
explanation — just names.

HARD RULES — these override every other instruction in this prompt:
- Do NOT name a disease in the summary unless BOTH analyze_literature AND analyze_clinical_trials
  ran successfully for that disease in this run. Mechanism evidence alone is not sufficient,
  even when an APPROVAL RELATIONSHIP is obvious.
- Every value in the summary must trace to a tool artifact you received this run. Do not
  estimate, round, or recall numbers from training knowledge. If you don't have a value, omit
  the field — never invent one.
- If a candidate's only tool call was REJECTED, exclude it from the summary entirely.
- No free-form sentences. No paragraph prose. Stick to the format above."""


def build_supervisor_agent(llm, svc, db):
    """Return (compiled supervisor agent, get_merged_allowlist).

    get_merged_allowlist is a zero-arg callable that snapshots the closure-scoped
    competitor + mechanism disease allowlist after the agent has finished running.
    run_supervisor_agent uses it to assemble findings against the merged view rather
    than reconstructing an allowlist from tool messages.
    """
    tools, get_merged_allowlist = build_supervisor_tools(llm=llm, svc=svc, db=db)
    agent = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
    return agent, get_merged_allowlist


async def run_supervisor_agent(
    agent,
    get_merged_allowlist,
    drug_name: str,
) -> SupervisorOutput:
    """Invoke the supervisor and assemble a SupervisorOutput from the run.

    Filters out tool calls that were rejected by the candidate guard, and canonicalises disease
    names against the merged competitor + mechanism allowlist so that casing variants (e.g.
    "Parkinson disease" vs "parkinson disease") do not produce duplicate findings, and so that
    mechanism-promoted diseases reach the findings list with their correct source tag.
    """
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Find repurposing opportunities for {drug_name}")
            ]
        }
    )

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

    # First pass: capture mechanism artifact and the supervisor's final summary.
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "analyze_mechanism":
            mechanism = msg.artifact
        elif msg.name == "finalize_supervisor":
            summary = msg.artifact or ""

    # Single source of truth: the merged allowlist the runtime tools enforced. Keyed by
    # lowercase disease name → (canonical_name, source). Source is "competitor", "mechanism",
    # or "both".
    allowed_lower = get_merged_allowlist()

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

    # Candidates surfaced to downstream consumers = every disease in the merged allowlist,
    # mapped back to its canonical name. Includes mechanism-promoted diseases.
    candidates = [canonical for (canonical, _) in allowed_lower.values()]

    return SupervisorOutput(
        drug_name=drug_name,
        candidates=candidates,
        mechanism=mechanism,
        findings=list(findings_by_disease.values()),
        summary=summary,
    )