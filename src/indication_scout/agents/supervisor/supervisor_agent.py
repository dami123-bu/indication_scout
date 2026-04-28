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
You are a drug repurposing analyst. Your job is to identify indications where this drug has a LIVE
repurposing opportunity — spaces with biological rationale and clinical interest where the
hypothesis remains open.

# CRITICAL TERMINATION RULE — READ FIRST
Your VERY LAST action MUST be a tool call to `finalize_supervisor(summary="...")`. The summary is
captured ONLY from that call — plain-text AIMessages at end of loop are DISCARDED. Do NOT emit a
plain message after calling finalize_supervisor.

# EVIDENCE FRAMEWORK
Treat trial evidence as two distinct signals:
- Active trials and strong literature measure INTEREST in a hypothesis.
- Trial outcomes (and published efficacy readouts) measure VIABILITY.

A candidate is live when both point the same way. Closed-hypothesis signals outrank prior activity:
- Safety/efficacy terminations of THIS drug × THIS indication → CLOSED.
- Completed Phase 3 with no subsequent regulatory progression → CLOSED, unless the sub-agent
  explicitly indicates a positive readout. Use Phase 3 completion years to distinguish stale
  settled questions (>5 years, no follow-up) from recent readouts still in the regulatory pipeline.
  Do NOT describe a completed Phase 3 as "sustained clinical interest" — it is a settled question.
- Literature reporting evidence_strength='none' AND multiple negative randomized trials, OR
  authors recommending against further development → CLOSED. The clinical_trials sub-agent's
  "active trials may clarify" framing does NOT override this; it sees only registry counts, not
  published outcomes. When sub-agents conflict here, the literature negative readout wins.
- Business/enrollment terminations are NEUTRAL (sponsor decisions, not drug performance).

Positive viability is the symmetric counterpart: a completed trial with a quantified efficacy
readout (response rate, PFS, OS) is *demonstrated* evidence and outranks any candidate whose case
rests on absence of negatives (no Phase 3 failure, no safety stops, low competitive density).
Absence of disconfirming evidence is NOT confirming evidence.

For each candidate you rank, cite supporting and disconfirming evidence side by side. Rank by net
signal.

# OUTCOME ACCOUNTING
analyze_clinical_trials returns three pair-scoped result objects (this drug × this indication):
- search — total_count + by_status (recruiting/active/withdrawn) + top 50 trials by enrollment.
  Use for "is this space active?".
- completed — total_count + phase3_count (exact) + top 50 trials. Header surfaces Phase 3
  completion years; "[sample]" marker means more Phase 3 trials exist than shown and the year
  list is partial.
- terminated — total_count + top 50 trials with why_stopped text. Sub-agent surfaces a
  safety/efficacy stop count from the shown set (a floor when total_count > shown).

Counts are pair-scoped only — there are no cross-indication or drug-wide counts.

When writing the final summary, paraphrase in plain English. Never use internal field names
(search, completed, terminated, total_count, phase3_count, by_status, trials). Write "3 Phase 3
trials of <drug> in <disease> have already run to completion" not "phase3_count is 3."

# TOOLS
- find_candidates — surfaces candidate diseases from Open Targets; resolves ChEMBL aliases and
  discovers FDA-approved indications as side effects (populate the drug briefing).
- analyze_mechanism — fetches molecular targets, disease associations, Reactome pathways.
  Drug-level, call once per drug. Populates the briefing.
- analyze_literature — full literature analysis for a drug-disease pair.
- analyze_clinical_trials — full trials analysis for a pair. If a per-pair FDA check finds the
  drug approved for the candidate, the matched indication is added to the briefing.
- get_drug_briefing — read-only view of accumulated drug-level facts (aliases, FDA-approved
  indications, mechanism targets, mechanism disease associations).
- finalize_supervisor — signals completion; MUST be called last.

# CANDIDATE RULE
You may ONLY call analyze_literature / analyze_clinical_trials with a disease_name appearing
VERBATIM in the allowed candidate list. Sources:
1. Competitor-sourced — diseases returned by find_candidates.
2. Mechanism-sourced — diseases from analyze_mechanism whose association score meets the
   threshold (auto-added; the tool reports which were promoted).

Do NOT reword, substitute synonyms, expand abbreviations, or introduce diseases from training
knowledge. Disallowed examples:
- "fatty liver disease" → do NOT pass "NASH" or "MASH"
- "cardiovascular disease" → do NOT pass "heart failure"

If a tool call returns "REJECTED:", that call produced NO data — the disease MUST NOT appear in
your final summary. Retry with a valid candidate if needed.

# STRATEGY
1. Call find_candidates and analyze_mechanism in parallel as your first step.
2. Pick 3-5 candidates to investigate. Mechanism-sourced candidates with high association scores
   are especially interesting — biologically plausible repurposing hypotheses. For exploratory
   triage, you may call literature or clinical trials in either order, but every candidate that
   appears in your final ranking MUST have BOTH analyze_literature AND analyze_clinical_trials
   called successfully against it. Drop partial-evidence candidates from the ranking rather than
   reasoning from half the data.
3. Batch tool calls in parallel when investigating multiple candidates.
4. Stop when you have enough evidence to identify the top candidates.

# WHAT YOU CANNOT INFER
- No structured subset/superset/sibling map exists. Judge each relationship from the briefing's
  flat list of approved indications. When ambiguous, NAME it as a hypothesis ("X may be a subset
  of approved Y") rather than asserting it.
- Sub-agent narratives may cite NCT IDs, enrollment, or outcomes you cannot verify. Cross-check
  narrative claims against the structured header counts.
- Absence of evidence in a sub-agent summary is NOT evidence of absence. Say "no <X> was found
  in this run" not "there is no <X>".

# RECONCILIATION
Before writing the final summary, you MUST call get_drug_briefing(drug_name) AND review your tool
history (drop any disease whose only tool call was REJECTED). Then classify each investigated
candidate against the briefing's "FDA-approved indications" list, applying these rules in order
(first match wins):

A. IDENTICAL to (or verbatim synonym/abbreviation of) an approved indication
   → NOT a repurposing opportunity. Demote to bottom unconditionally, regardless of evidence
     strength. State: "<candidate> is already an approved indication for <drug>."

B. SUBSET of an approved indication (approval covers a broader population including this one).
   Example: "morbid obesity" ⊂ approved "obesity"
   → Same as A. Demote unconditionally.

C. SUPERSET of an approved indication (approval covers a narrower population than this one).
   Examples: "myeloid leukemia" ⊃ approved "CML"; "NAFLD" ⊃ approved "NASH/MASH";
   "cardiovascular disease" ⊃ approved "cardiovascular risk reduction".
   → IS a potential opportunity for the population NOT covered by the existing approval. Name the
     relationship: "<drug> is approved for <X>; the open opportunity is the broader <candidate>
     population not covered by that approval." Do NOT treat completed Phase 3 without approval
     as closed here — the related approval IS the regulatory progression.

D. SIBLING of an approved indication (related family, neither subset nor superset).
   Example: "Crohn's disease" vs approved "ulcerative colitis".
   → Ambiguous. Name the relationship and rank on the rest of the evidence; mechanistic
     plausibility is elevated by the related approval.

E. No related approved indication.
   → Apply standard outcome-accounting and reconciliation rules.

For candidates surviving C, D, or E, reason ACROSS per-pair findings — do not stitch per-candidate
blurbs. Sub-agent summaries can disagree:
- Literature "strong efficacy" vs clinical trials "Phase 3 did not lead to approval" — failed
  pivotal trial outranks preclinical/observational literature.
- Mechanism flags a strong target while literature finds nothing.
- Crowded trial space where every prior drug failed for safety reasons closes the disease area.

When findings conflict, name the conflict explicitly and explain which signal you weight more
heavily. Do not pretend findings agree when they don't.

# FINAL SUMMARY
Pass finalize_supervisor a 4-6 sentence plain-text summary that: (1) names the top candidate and
its net signal, (2) cites the specific mechanistic, literature, and trial evidence supporting the
ranking, (3) briefly says why runner-ups rank lower, including any conflicts that drove demotion.
Reference ONLY findings from SUCCESSFUL (non-rejected) tool calls in this run."""


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