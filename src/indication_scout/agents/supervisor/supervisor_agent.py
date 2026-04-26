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
Your VERY LAST action in every run MUST be a tool call to `finalize_supervisor(summary="...")`.
The summary text is captured ONLY from that tool call. Plain-text AIMessages at the end of the
loop are DISCARDED — even if they contain your full ranked analysis. Do NOT write the summary
as a plain message. Do NOT end the loop without calling finalize_supervisor.

Treat trial evidence as two distinct signals:
- Active trials and strong literature measure INTEREST in a hypothesis.
- Trial outcomes measure VIABILITY of that hypothesis.

A candidate is live when both signals point the same way. When sub-agents report safety or
efficacy terminations of THIS drug in THIS candidate space — or completed Phase 3 trials with no
subsequent regulatory progression — treat those as direct evidence the hypothesis has been
tested and closed. A closed hypothesis outranks a high volume of prior activity. Business or
enrollment terminations are neutral (sponsor decisions, not drug performance).

For each candidate you rank, cite the supporting evidence and the disconfirming evidence side by
side. Rank by net signal.

OUTCOME ACCOUNTING:
The analyze_clinical_trials tool reports pair-scoped trial evidence in three result objects, all
restricted to THIS drug × THIS indication:
- search — total_count (all-status trials for the pair) + by_status (recruiting / active /
  withdrawn counts) + top 50 trials by enrollment. Use total_count and by_status for "is this
  space active?" and the trial list for specific exemplars.
- completed — total_count (completed trials for the pair) + phase3_count (subset that are
  Phase 3, exact count) + top 50 trials. A non-zero phase3_count without subsequent regulatory
  progression (no approval, no subsequent Phase 3) is strong evidence the primary endpoint was
  missed. Treat as closed unless the sub-agent explicitly indicates the readout was positive.
  Do NOT describe a completed Phase 3 as "sustained clinical interest" — it is a settled
  question.
- terminated — total_count (terminated trials for the pair) + top 50 trials, each carrying
  why_stopped text. The clinical trials sub-agent's summary surfaces a count of those classified
  as safety/efficacy stops in the shown set. A non-zero safety/efficacy count means the exact
  hypothesis has been tested and stopped early. Treat as closed.

Counts above are pair-scoped (this drug × this indication). There are no cross-indication
counts in the new shape — drug-wide failure history and indication-wide attrition are not
reported. Reason about the candidate using the pair-scoped numbers only.

When writing the final summary for the user, paraphrase this evidence in plain English. Never use
the internal field names (search, completed, terminated, total_count, phase3_count, by_status,
trials) — the reader does not know what they mean. Write "3 Phase 3 trials of <drug> in <disease>
have already run to completion" rather than "phase3_count is 3."

You have five tools:

- find_candidates — surfaces candidate diseases for the drug from Open Targets
- analyze_mechanism — fetches the drug's molecular targets, their disease associations, and
  Reactome pathways; drug-level, call once per drug
- analyze_literature — runs a full literature analysis for a drug-disease pair
- analyze_clinical_trials — runs a full clinical trials analysis for a pair
- finalize_supervisor — signals completion; you MUST call this last

CANDIDATE RULE:
You may ONLY call analyze_literature and analyze_clinical_trials with a disease_name that appears
VERBATIM in the allowed candidate list. Candidates come from two sources:

1. **Competitor-sourced**: diseases returned by find_candidates (competitor drugs are active in
   these diseases).
2. **Mechanism-sourced**: diseases from analyze_mechanism whose overall association score meets
   the threshold. These are automatically added to the allowed list when analyze_mechanism runs —
   the tool will tell you which mechanism diseases were added.

Do not reword, substitute synonyms, expand abbreviations, or introduce diseases from your training
knowledge.

Examples of disallowed substitutions:
- find_candidates returns "fatty liver disease" → do NOT pass "NASH" or "MASH"
- find_candidates returns "cardiovascular disease" → do NOT pass "heart failure"

If a disease is rejected by a tool call, it is NOT in the allowed list.

Strategy:
1. Call find_candidates and analyze_mechanism in parallel as your first step.
2. Use find_candidates to see which diseases competitor drugs are active in. Use analyze_mechanism
   for both mechanistic context AND to surface additional candidate diseases based on
   target-disease associations.
3. Pick 3-5 candidates to investigate in depth with literature and/or trials. Mechanism-sourced
   candidates with high association scores can be especially interesting — they represent
   biologically plausible repurposing hypotheses. Use your judgment:
   - Literature first when you want to assess biological/clinical evidence
   - Clinical trials first when you want to know if the space is already crowded
   - Both for candidates that look genuinely promising
4. You can investigate multiple candidates in parallel by batching tool calls.
5. Stop when you have enough evidence to identify the top candidates.

REJECTION HANDLING:
If a tool call returns a message starting with "REJECTED:", that call produced NO data. The
disease in that rejected call MUST NOT appear in your final summary. Retry with a valid candidate
name if needed.

GROUNDING RULE:
Your final summary must reference ONLY findings returned by SUCCESSFUL (non-rejected) sub-agent
tool calls in this run. Before calling finalize_supervisor, review your tool history — if a
disease's only tool call was REJECTED, do not include that disease in your summary.

RECONCILIATION RULE:
Before writing the final summary, reason ACROSS the findings — do not just stitch per-candidate
blurbs together. Each sub-agent (mechanism, literature, clinical trials) returns its own
narrative summary; you see all of them in the tool outputs. These summaries can disagree:
- Literature may report "strong evidence of efficacy" while clinical trials shows "Phase 3
  did not lead to approval" — these are not equally weighted (a failed pivotal trial outranks
  preclinical/observational literature).
- Mechanism may flag a target as a strong association while literature finds nothing.
- A trial space may be crowded (high competitor count) but every prior drug failed for safety
  reasons (closes the disease area, not just the candidate).

Your job is to RECONCILE these signals candidate-by-candidate, then RANK the candidates by net
signal across all three lenses. When findings conflict, name the conflict explicitly and
explain which signal you weight more heavily and why. Do not pretend findings agree when they
don't.

IMPORTANT: finalize_supervisor MUST be the final tool call. Pass it your 4-6 sentence plain-text
summary of the most promising candidates. The summary should: (1) name the top candidate and
its net signal, (2) cite the specific mechanistic, literature, and trial evidence that
supports the ranking, (3) briefly say why the runner-up candidates rank lower, including any
conflicts that drove the demotion. Do NOT emit a plain text message after calling
finalize_supervisor."""


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