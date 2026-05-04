"""Supervisor tools — wraps sub-agents as tools.

Each sub-agent (literature, clinical trials) becomes a single tool the supervisor can call. The
tool runs the full sub-agent and returns its typed output as an artifact, plus a short summary
string for the LLM.

There's also a find_candidates tool that hits Open Targets directly to surface disease candidates
for a drug.
"""

import asyncio
import logging
import time
from datetime import date
from typing import Literal

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.config import get_settings
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.data_sources.fda import FDAClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.services.approval_check import (
    get_approved_indications,
    get_fda_approved_disease_mapping,
    list_approved_indications_at,
    list_approved_indications_from_labels,
)

logger = logging.getLogger(__name__)

from indication_scout.agents._trial_formatting import (
    _borda_rank_by_enrollment_and_recency,
    _format_trial_table,
    _phase_distribution,
)
from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)
from indication_scout.agents.literature.literature_agent import (
    build_literature_agent,
    run_literature_agent,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_agent import (
    build_mechanism_agent,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.services.retrieval import RetrievalService


def build_supervisor_tools(
    llm,
    svc: RetrievalService,
    db: Session,
    date_before: date | None = None,
) -> tuple[list, "callable", "callable"]:
    """Build supervisor tools that close over the sub-agents.

    The literature and clinical trials agents are compiled once here and reused across calls — no
    need to rebuild them per invocation.

    `date_before` is forwarded to the literature and clinical trials sub-agents so all PubMed and
    ClinicalTrials.gov queries respect the same temporal cutoff. Mechanism sub-agent does not
    accept it (OpenTargets has no date-filtering API).

    Returns (tools, get_merged_allowlist) where get_merged_allowlist() snapshots the post-merge
    competitor + mechanism disease allowlist (lowercase name → (canonical_name, source)). The
    snapshot reflects whatever state the closure holds at call time — intended to be read after
    the agent loop has finished.
    """

    # Build sub-agents once at supervisor construction (except literature — see below)
    ct_agent = build_clinical_trials_agent(llm=llm, date_before=date_before)
    mech_agent = build_mechanism_agent(llm=llm)

    # Closure-scoped allowlist — populated by find_candidates and analyze_mechanism, checked by
    # analyze_literature / analyze_clinical_trials.
    # allowed_diseases: lowercase disease name → (canonical_name, source)
    allowed_diseases: dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]] = {}
    # EFO ID → lowercase disease name (key into allowed_diseases). Lets analyze_mechanism dedup
    # mechanism candidates against competitor entries by ontology ID even when names differ
    # (e.g. "NSCLC" vs "non-small cell lung cancer").
    allowed_efo_ids: dict[str, str] = {}
    # Seed-phase gates. analyze_literature and analyze_clinical_trials must not run until both
    # find_candidates and analyze_mechanism have populated the allowlist (and merged), otherwise
    # they may investigate a disease against a stale competitor-only or mechanism-only view.
    # Both events are set in a try/finally so a sub-agent crash doesn't deadlock downstream tools.
    find_candidates_done = asyncio.Event()
    analyze_mechanism_done = asyncio.Event()

    # Drug-level shared store. Populated by sub-agents as they run; surfaced
    # to the supervisor via get_drug_briefing. Keyed by drug name (normalized).
    # See supervisor_ideas.md for design rationale.
    drug_facts: dict[str, dict] = {}

    # Holdout-only: artifacts produced by investigate_top_candidates. The tool
    # invokes analyze_literature/analyze_clinical_trials directly (not through
    # the LangGraph ReAct loop), so their tool messages don't reach
    # result["messages"]. We stash them here and run_supervisor_agent reads
    # them via get_auto_findings() after the agent run completes.
    # Keyed by lowercase canonical disease name → {"literature": LiteratureOutput,
    # "clinical_trials": ClinicalTrialsOutput}.
    auto_findings: dict[str, dict] = {}

    def _drug_key(drug_name: str) -> str:
        return drug_name.lower().strip()

    def _ensure_drug_entry(drug_name: str) -> dict:
        key = _drug_key(drug_name)
        if key not in drug_facts:
            drug_facts[key] = {
                "drug_name": key,
                "drug_aliases": [],               # ChEMBL trade/generic names
                "approved_indications": [],       # list of indication strings
                "mechanism_targets": [],          # list of (gene, action_type)
                "mechanism_disease_associations": [],  # list of (gene, disease, score)
            }
        return drug_facts[key]

    def _render_briefing(drug_name: str) -> str:
        """Render drug_facts[drug_name] as a markdown briefing."""
        entry = drug_facts.get(_drug_key(drug_name))
        if entry is None:
            return f"DRUG INTAKE: {drug_name}\n- (no facts collected yet)"

        lines = [f"DRUG INTAKE: {entry['drug_name']}"]

        if entry["drug_aliases"]:
            lines.append(f"- Trade/generic names: {', '.join(entry['drug_aliases'])}")
        else:
            lines.append("- Trade/generic names: (not yet resolved)")

        if entry["approved_indications"]:
            lines.append("- FDA-approved indications:")
            for ind in entry["approved_indications"]:
                lines.append(f"  - {ind}")
        else:
            lines.append("- FDA-approved indications: (none discovered in this run)")

        if entry["mechanism_targets"]:
            target_strs = [f"{g} ({a})" for g, a in entry["mechanism_targets"]]
            lines.append(f"- Targets: {', '.join(target_strs)}")
        else:
            lines.append("- Targets: (mechanism agent has not run)")

        if entry["mechanism_disease_associations"]:
            lines.append("- Top mechanism-disease associations:")
            # cap at 10 to keep briefing terse
            for g, d, s in entry["mechanism_disease_associations"][:10]:
                # Hide score when it's the placeholder (not surfaced by
                # MechanismCandidate). Show otherwise.
                score_str = f" (score {s:.2f})" if s > 0 else ""
                lines.append(f"  - {g} → {d}{score_str}")

        return "\n".join(lines)

    @tool(response_format="content_and_artifact")
    async def find_candidates(drug_name: str) -> tuple[str, list[str]]:
        """Surface candidate diseases for repurposing this drug.

        Uses Open Targets to find diseases where competitor drugs (drugs sharing the same molecular
        targets) are being developed. Returns a list of disease names ranked by competitor activity.
        """
        try:
            return await _find_candidates_impl(drug_name)
        finally:
            # Always release the seed gate so a failure here doesn't deadlock analyze_literature
            # / analyze_clinical_trials. They will see an empty allowlist and reject downstream.
            find_candidates_done.set()

    async def _find_candidates_impl(drug_name: str) -> tuple[str, list[str]]:
        drug_name = normalize_drug_name(drug_name)
        chembl_id = await resolve_drug_name(drug_name, svc.cache_dir)
        competitors = await svc.get_drug_competitors(chembl_id, date_before=date_before)
        diseases = list(competitors.keys())

        # Drug-level intake: populate the shared store with aliases and any
        # FDA-approved indications discovered during the candidate filter.
        entry = _ensure_drug_entry(drug_name)
        try:
            entry["drug_aliases"] = await get_all_drug_names(chembl_id, svc.cache_dir)
        except Exception as e:
            logger.warning("find_candidates: get_all_drug_names failed for %s: %s", chembl_id, e)

        # Seed approved_indications from the drug's own FDA label, independent of
        # any candidate list. Without this, an approved indication that doesn't
        # appear among OpenTargets competitor diseases (e.g. semaglutide × MASH)
        # never reaches the briefing and the supervisor cannot reason about
        # subset/superset relationships against it.
        #
        # When date_before is set, swap the live openFDA path for the hardcoded
        # approvals table — the live path leaks today's approvals into a
        # holdout. Drugs not in the table return [] and approval reasoning is
        # silently disabled for that holdout run (see PLAN_date_before.md).
        seed_aliases = entry["drug_aliases"] or [drug_name]
        try:
            if date_before is not None:
                seeded = list_approved_indications_at(drug_name, date_before)
            else:
                async with FDAClient(cache_dir=svc.cache_dir) as fda_client:
                    label_texts = await fda_client.get_all_label_indications(seed_aliases)
                seeded = await list_approved_indications_from_labels(
                    label_texts=label_texts,
                    cache_dir=svc.cache_dir,
                )
            if seeded:
                existing = {ind.lower().strip() for ind in entry["approved_indications"]}
                for ind in seeded:
                    if ind.lower().strip() not in existing:
                        entry["approved_indications"].append(ind)
                # logger.warning(
                #     "[TOOL] find_candidates seeded %d approved indication(s) from %s: %s",
                #     len(seeded),
                #     "hardcoded table" if date_before is not None else "label",
                #     seeded,
                # )
        except Exception as e:
            logger.warning(
                "find_candidates: approved-indication seed failed for %s: %s",
                drug_name, e,
            )

        # Drop competitor diseases already approved for this drug. Same swap as
        # above: hardcoded table when date_before is set, live FDA otherwise.
        fda_approved_lower: set[str] = set()
        if diseases:
            if date_before is not None:
                fda_approved = get_approved_indications(
                    drug_name=drug_name,
                    candidate_diseases=diseases,
                    as_of=date_before,
                )
            else:
                mapping = await get_fda_approved_disease_mapping(
                    drug_name=drug_name,
                    candidate_diseases=diseases,
                    cache_dir=svc.cache_dir,
                )
                fda_approved = {disease for disease, is_approved in mapping.items() if is_approved}
            if fda_approved:
                logger.warning(
                    "[TOOL] find_candidates FDA approval check removing %d competitor diseases "
                    "(source: %s): %s",
                    len(fda_approved),
                    "hardcoded table" if date_before is not None else "live FDA",
                    fda_approved,
                )
                # Record the approved indications in the shared store. These were
                # discovered as side effect of candidate filtering — even though
                # they're dropped from the candidate list, the supervisor needs
                # to see them to reason about subset/superset relationships
                # (e.g. CML approval makes "myeloid leukemia" candidate ambiguous).
                existing = {ind.lower().strip() for ind in entry["approved_indications"]}
                for ind in fda_approved:
                    if ind.lower().strip() not in existing:
                        entry["approved_indications"].append(ind)
            fda_approved_lower = {d.lower().strip() for d in fda_approved}

        diseases = [d for d in diseases if d.lower().strip() not in fda_approved_lower]

        # Cap how many candidates the supervisor sees. Order is preserved from upstream
        # ranking (OpenTargets competitor merge), so this keeps the top-N.
        candidate_cap = get_settings().supervisor_candidate_cap
        if len(diseases) > candidate_cap:
            logger.warning(
                "[TOOL] find_candidates capping candidates from %d to %d (SUPERVISOR_CANDIDATE_CAP)",
                len(diseases),
                candidate_cap,
            )
            diseases = diseases[:candidate_cap]

        allowed_diseases.clear()
        allowed_efo_ids.clear()
        for d in diseases:
            allowed_diseases[d.lower().strip()] = (d, "competitor")

        # Pull EFO IDs for the competitor allowlist directly from the raw OT cache. Used by
        # analyze_mechanism to dedup mechanism candidates against competitor entries by ontology
        # ID. Disease names that don't resolve to an EFO (e.g. renamed by the LLM merge step)
        # simply don't get an entry — analyze_mechanism falls back to name match in that case.
        async with OpenTargetsClient(cache_dir=svc.cache_dir) as ot_client:
            raw = await ot_client.get_drug_competitors(chembl_id, date_before=date_before)
        for disease_lower in allowed_diseases:
            efo_id = raw["disease_efo_ids"].get(disease_lower)
            if efo_id:
                allowed_efo_ids[efo_id] = disease_lower

        # logger.warning("[TOOL] find_candidates(%r [%s]) -> %s", drug_name, chembl_id, diseases)
        return (
            f"Found {len(diseases)} candidate diseases for {drug_name} ({chembl_id})",
            diseases,
        )

    def _reject(disease_name: str, tool_label: str, empty_output):
        valid = sorted(allowed_diseases.keys())
        msg = (
            f"REJECTED: '{disease_name}' is not in the allowed candidate list. "
            f"You must call {tool_label} only with a disease_name returned VERBATIM by "
            f"find_candidates or added from mechanism associations. "
            f"Do not reword, substitute synonyms, or introduce diseases from training knowledge. "
            f"Valid candidates: {valid}"
        )
        #logger.warning("[TOOL] %s REJECTED disease=%r", tool_label, disease_name)
        return msg, empty_output

    @tool(response_format="content_and_artifact")
    async def analyze_literature(
        drug_name: str, disease_name: str
    ) -> tuple[str, LiteratureOutput]:
        """Run a full literature analysis for a drug-disease pair.

        Investigates published evidence via PubMed, embeds and re-ranks abstracts, and produces a
        structured evidence summary with strength rating (none / weak / moderate / strong).
        """
        # Wait for both seed tools to finish populating the allowlist. Without this, parallel
        # tool calls can hit analyze_literature before find_candidates / analyze_mechanism have
        # merged their candidates, causing legitimate diseases to be rejected.
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)
        # Build a fresh agent per call so the closure-scoped store dict in literature_tools is not
        # shared across disease invocations.
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(disease_name, "analyze_literature", LiteratureOutput())

        #logger.warning("[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name)


        lit_agent = build_literature_agent(llm=llm, svc=svc, db=db, date_before=date_before)
        t0 = time.perf_counter()
        #logger.warning("[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name)

        output = await run_literature_agent(lit_agent, drug_name, disease_name)
        logger.info("analyze_literature took %.2fs for %s × %s", time.perf_counter() - t0, drug_name, disease_name)
        strength = (
            output.evidence_summary.strength
            if output.evidence_summary
            else "no data"
        )
        header = (
            f"Literature for {drug_name} × {disease_name}: "
            f"{len(output.pmids)} PMIDs, strength={strength}."
        )
        sub_agent_summary = output.summary or ""
        summary = (
            f"{header}\n\n{sub_agent_summary}".strip()
            if sub_agent_summary
            else header
        )
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_clinical_trials(
        drug_name: str, disease_name: str
    ) -> tuple[str, ClinicalTrialsOutput]:
        """Run a full clinical trials analysis for a drug-disease pair.

        Checks ClinicalTrials.gov for existing trials, competitive landscape, and terminated
        trials (safety/efficacy red flags).
        """
        # Wait for both seed tools to finish populating the allowlist (see analyze_literature).
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(disease_name, "analyze_clinical_trials", ClinicalTrialsOutput())

        #logger.warning("[TOOL] analyze_clinical_trials(drug=%r, disease=%r)", drug_name, disease_name)
        t0 = time.perf_counter()
        output = await run_clinical_trials_agent(ct_agent, drug_name, disease_name)
        #logger.info("analyze_clinical_trials took %.2fs for %s × %s", time.perf_counter() - t0, drug_name, disease_name)

        # Drug-level write-through: when the FDA check matches the candidate
        # against an approved indication, capture it in the supervisor's
        # briefing so subsequent reasoning sees the approval status. Trial
        # data still flows through to the summary below — the sub-agent
        # always investigates fully now (no short-circuits).
        approval = output.approval
        if approval is not None and approval.is_approved:
            entry = _ensure_drug_entry(drug_name)
            matched = approval.matched_indication or disease_name
            existing = {ind.lower().strip() for ind in entry["approved_indications"]}
            if matched.lower().strip() not in existing:
                entry["approved_indications"].append(matched)

        # Normal path: counts come from the new exact-count tools (countTotal
        # API). Each scope owns its own count; no cross-scope summing.
        search = output.search
        completed = output.completed
        terminated = output.terminated

        n_total = search.total_count if search else 0
        n_recruiting = search.by_status.get("RECRUITING", 0) if search else 0
        n_active_not_recruiting = (
            search.by_status.get("ACTIVE_NOT_RECRUITING", 0) if search else 0
        )
        n_withdrawn = search.by_status.get("WITHDRAWN", 0) if search else 0
        n_completed = completed.total_count if completed else 0
        n_terminated = terminated.total_count if terminated else 0
        # safety/efficacy classification is computed from the top-50 shown
        # terminated trials; if total_count > len(trials) this is a floor.
        n_safety_efficacy_shown = (
            sum(
                1 for t in terminated.trials
                if _classify_stop_reason(t.why_stopped) in {"safety", "efficacy"}
            )
            if terminated else 0
        )
        header = (
            f"Clinical trials for {drug_name} × {disease_name}: "
            f"{n_total} total ({n_recruiting} recruiting, "
            f"{n_active_not_recruiting} active, {n_withdrawn} withdrawn). "
            f"{n_completed} completed. "
            f"{n_terminated} terminated "
            f"({n_safety_efficacy_shown} safety/efficacy in shown set)."
        )

        completed_trials = completed.trials if completed else []
        terminated_trials = terminated.trials if terminated else []

        completed_phase_dist = _phase_distribution(completed_trials)
        terminated_phase_dist = _phase_distribution(terminated_trials)

        completed_top = _borda_rank_by_enrollment_and_recency(completed_trials, k=10)
        terminated_top = _borda_rank_by_enrollment_and_recency(terminated_trials, k=10)

        completed_table = _format_trial_table(
            completed_top,
            columns=(
                "nct_id",
                "phase",
                "start_date",
                "completion_date",
                "mesh",
                "title",
            ),
            cap=10,
        )
        terminated_table = _format_trial_table(
            terminated_top,
            columns=(
                "nct_id",
                "phase",
                "stop_reason",
                "start_date",
                "completion_date",
                "mesh",
                "title",
            ),
            cap=10,
            include_why_stopped=True,
            stop_classifier=_classify_stop_reason,
        )

        structured = (
            f"\n\nPhase distribution (completed): {completed_phase_dist}\n"
            f"Phase distribution (terminated): {terminated_phase_dist}\n\n"
            f"Completed trials (top 10 by enrollment + recency):\n"
            f"{completed_table}\n\n"
            f"Terminated trials (top 10 by enrollment + recency):\n"
            f"{terminated_table}"
        )

        sub_agent_summary = output.summary or ""
        summary = f"{header}{structured}"
        if sub_agent_summary:
            summary = f"{summary}\n\n{sub_agent_summary}"
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_mechanism(drug_name: str) -> tuple[str, MechanismOutput]:
        """Run the mechanism sub-agent for a drug.

        The mechanism agent returns target-level MoA data and the agent's narrative summary.
        Mechanism-surfaced candidates are promoted into the investigation allowlist so
        analyze_literature / analyze_clinical_trials can investigate them downstream.
        """
        try:
            return await _analyze_mechanism_impl(drug_name)
        finally:
            # Always release the seed gate so a failure here doesn't deadlock analyze_literature
            # / analyze_clinical_trials.
            analyze_mechanism_done.set()

    async def _analyze_mechanism_impl(drug_name: str) -> tuple[str, MechanismOutput]:
        drug_name = normalize_drug_name(drug_name)
        output = await run_mechanism_agent(mech_agent, drug_name)
        # logger.warning("[TOOL] analyze_mechanism(drug=%r)", drug_name)

        # The mechanism sub-agent can run in parallel with find_candidates, but the merge step
        # below must observe a fully-populated competitor allowlist. Wait here so we don't dedup
        # mechanism candidates against an empty/partial competitor list.
        await find_candidates_done.wait()

        promoted: list[str] = []
        async with OpenTargetsClient(cache_dir=svc.cache_dir) as ot_client:
            for candidate in output.candidates:
                key = candidate.disease_name.lower().strip()
                if not key:
                    continue

                # Three-step dedup against the competitor allowlist.
                #   1. ID match — common case when both sides emit the same OT canonical ID.
                #   2. Exact-name match — covers the case where one side lacks an ID.
                #   3. OT name-resolve — when the candidate's raw ID and name both miss, ask
                #      OT's search to canonicalize the name to its disease ID and retry the ID
                #      match. Catches cross-ontology drift (EFO vs MONDO) and synonyms.
                existing_key: str | None = None
                if candidate.disease_id and candidate.disease_id in allowed_efo_ids:
                    existing_key = allowed_efo_ids[candidate.disease_id]
                elif key in allowed_diseases:
                    existing_key = key
                else:
                    resolved_id = await ot_client.resolve_disease_id(candidate.disease_name)
                    if resolved_id and resolved_id in allowed_efo_ids:
                        existing_key = allowed_efo_ids[resolved_id]

                if existing_key is not None:
                    existing_name, source = allowed_diseases[existing_key]
                    if source == "competitor":
                        allowed_diseases[existing_key] = (existing_name, "both")
                    # Record the disease ID against the existing row when we learned it from
                    # this mechanism candidate (e.g. competitor entry had no ID). Improves
                    # dedup for subsequent candidates in the same run.
                    if candidate.disease_id and candidate.disease_id not in allowed_efo_ids:
                        allowed_efo_ids[candidate.disease_id] = existing_key
                else:
                    allowed_diseases[key] = (candidate.disease_name, "mechanism")
                    if candidate.disease_id:
                        allowed_efo_ids[candidate.disease_id] = key
                    promoted.append(candidate.disease_name)

        if promoted:
            logger.warning(
                "[TOOL] analyze_mechanism promoted %d mechanism-only candidates to allowlist: %s",
                len(promoted), promoted,
            )

        # Drug-level write-through: populate mechanism_targets and
        # mechanism_disease_associations in the shared store. Captured per-MoA
        # so the briefing can show "ABL1 (INHIBITOR), KIT (INHIBITOR)".
        entry = _ensure_drug_entry(drug_name)
        target_pairs: list[tuple[str, str]] = []
        seen_target_pairs: set[tuple[str, str]] = set()
        for moa in output.mechanisms_of_action:
            for sym in moa.target_symbols:
                pair = (sym, moa.action_type or "UNKNOWN")
                if pair not in seen_target_pairs:
                    seen_target_pairs.add(pair)
                    target_pairs.append(pair)
        entry["mechanism_targets"] = target_pairs

        # Mechanism candidates already carry the high-score target→disease
        # associations the agent surfaced. We don't have the raw scores on the
        # candidate model — record the pair without a score for now.
        assocs: list[tuple[str, str, float]] = []
        seen_assoc_pairs: set[tuple[str, str]] = set()
        for cand in output.candidates:
            pair_key = (cand.target_symbol, cand.disease_name)
            if pair_key in seen_assoc_pairs:
                continue
            seen_assoc_pairs.add(pair_key)
            # Score not surfaced on MechanismCandidate; use 0.0 as a placeholder.
            # The supervisor only needs to know "this gene is associated with
            # this disease per OT mechanism evidence" — the briefing renderer
            # will hide the score if it's the placeholder.
            assocs.append((cand.target_symbol, cand.disease_name, 0.0))
        entry["mechanism_disease_associations"] = assocs

        n_targets = len(output.drug_targets)
        header = (
            f"Mechanism analysis for {drug_name}: {n_targets} targets, "
            f"{len(output.candidates)} mechanism candidates "
            f"({len(promoted)} new to allowlist)."
        )
        sub_agent_summary = output.summary or ""
        summary = (
            f"{header}\n\n{sub_agent_summary}".strip()
            if sub_agent_summary
            else header
        )
        return summary, output

    @tool
    def get_drug_briefing(drug_name: str) -> str:
        """Return the accumulated drug-level briefing for this drug.

        Read-only view of facts collected by find_candidates, analyze_mechanism,
        and analyze_clinical_trials during this run: ChEMBL aliases, FDA-approved
        indications discovered, mechanism targets, and mechanism disease
        associations. Call this before finalize_supervisor to check whether any
        candidate is related to an approved indication (subset/superset/sibling).
        """
        drug_name = normalize_drug_name(drug_name)
        return _render_briefing(drug_name)

    # Holdout-only tool: bulk-investigate the top-N candidates with no LLM
    # discretion. The probe (scripts/probe_supervisor_t2dm.py) showed the
    # supervisor LLM systematically skips "obvious" candidates like T2DM for
    # semaglutide regardless of prompt instructions. In holdout mode that's
    # exactly the candidate the holdout is testing, so we remove the LLM's
    # ability to skip by auto-investigating the top-10.
    HOLDOUT_INVESTIGATION_CAP = 10

    @tool(response_format="content_and_artifact")
    async def investigate_top_candidates(
        drug_name: str,
    ) -> tuple[str, list[dict]]:
        """[HOLDOUT MODE ONLY] Auto-investigate the top-10 candidates from the merged allowlist.

        Runs analyze_literature AND analyze_clinical_trials in parallel for the top 10
        candidates by mechanism+competitor strength. Removes the LLM's ability to skip
        "obvious" candidates that holdout-mode evaluations specifically need to recover.

        Call this ONCE, after find_candidates and analyze_mechanism complete. After this
        returns, you may still call analyze_literature / analyze_clinical_trials for
        candidates beyond the top 10 if you want.
        """
        # Wait for both seed tools to finish populating the allowlist.
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)

        # Top-N from the merged allowlist. Insertion order preserves
        # find_candidates's competitor ranking, with mechanism-promoted
        # entries appended in the order analyze_mechanism processed them.
        top_n = list(allowed_diseases.items())[:HOLDOUT_INVESTIGATION_CAP]
        if not top_n:
            return "No candidates in allowlist; nothing to investigate.", []

        canonical_diseases = [canonical for _, (canonical, _) in top_n]
        # logger.warning(
        #     "[TOOL] investigate_top_candidates auto-investigating %d candidates: %s",
        #     len(canonical_diseases), canonical_diseases,
        # )

        # Fan out: analyze_literature + analyze_clinical_trials in parallel.
        # Pass a ToolCall-shaped dict (not a plain args dict) so .ainvoke()
        # returns a ToolMessage with .artifact populated. A plain dict input
        # returns just the content string and we lose the typed artifact.
        async def _invest(disease: str) -> tuple[str, dict]:
            disease_slug = disease.lower().replace(" ", "_")
            lit_call = analyze_literature.ainvoke(
                {
                    "name": "analyze_literature",
                    "args": {"drug_name": drug_name, "disease_name": disease},
                    "id": f"auto_lit_{disease_slug}",
                    "type": "tool_call",
                }
            )
            ct_call = analyze_clinical_trials.ainvoke(
                {
                    "name": "analyze_clinical_trials",
                    "args": {"drug_name": drug_name, "disease_name": disease},
                    "id": f"auto_ct_{disease_slug}",
                    "type": "tool_call",
                }
            )
            lit_msg, ct_msg = await asyncio.gather(lit_call, ct_call)

            lit_artifact = lit_msg.artifact
            ct_artifact = ct_msg.artifact
            # Stash artifacts in the closure so run_supervisor_agent can
            # merge them into the SupervisorOutput. The LangGraph ReAct
            # loop doesn't see these tool messages because they were
            # invoked directly, not through the agent.
            auto_findings[disease.lower().strip()] = {
                "literature": lit_artifact,
                "clinical_trials": ct_artifact,
            }
            strength = (
                lit_artifact.evidence_summary.strength
                if lit_artifact and lit_artifact.evidence_summary
                else "no data"
            )
            n_pmids = len(lit_artifact.pmids) if lit_artifact else 0
            n_total = (
                ct_artifact.search.total_count
                if ct_artifact and ct_artifact.search
                else 0
            )
            n_completed = (
                ct_artifact.completed.total_count
                if ct_artifact and ct_artifact.completed
                else 0
            )
            n_terminated = (
                ct_artifact.terminated.total_count
                if ct_artifact and ct_artifact.terminated
                else 0
            )
            return disease, {
                "disease": disease,
                "literature_strength": strength,
                "literature_pmids": n_pmids,
                "trials_total": n_total,
                "trials_completed": n_completed,
                "trials_terminated": n_terminated,
            }

        results = await asyncio.gather(*(_invest(d) for d in canonical_diseases))
        artifacts = [a for _, a in results]

        # One-line-per-disease compact summary the LLM can rank against.
        lines = [
            f"Auto-investigated {len(artifacts)} top candidates "
            f"for {drug_name}:"
        ]
        for a in artifacts:
            lines.append(
                f"  - {a['disease']}: literature {a['literature_strength']}, "
                f"{a['literature_pmids']} PMIDs; trials {a['trials_total']} total, "
                f"{a['trials_completed']} completed, {a['trials_terminated']} terminated"
            )
        return "\n".join(lines), artifacts

    @tool(response_format="content_and_artifact")
    async def finalize_supervisor(
        summary: str, blurbs: list[dict] | None = None
    ) -> tuple[str, dict]:
        """Signal that the repurposing analysis is complete.

        Call this as the very last step. This terminates the agent loop.

        Arguments:
        - summary: your ranked structured fact list of investigated candidates
          (see WRITING THE SUMMARY in the system prompt).
        - blurbs: a list of {"disease": <name>, "blurb": <exactly 2 sentences>} entries
          for the TOP 5 ranked candidates in your summary, in rank order. Each
          blurb must synthesize ONLY the literature and clinical_trials sub-agent
          summaries you saw for that disease this run — do not include mechanism
          content. Pass an empty list if no candidates were investigated. The
          disease name must match a name returned verbatim by find_candidates
          or promoted by analyze_mechanism (otherwise the blurb is dropped).
        """
        validated: list[dict] = []
        for item in blurbs or []:
            disease = (item.get("disease") or "").strip()
            blurb = (item.get("blurb") or "").strip()
            if not disease or not blurb:
                continue
            if disease.lower().strip() not in allowed_diseases:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping blurb for disease=%r "
                    "(not in allowlist)",
                    disease,
                )
                continue
            validated.append({"disease": disease, "blurb": blurb})
        artifact = {"summary": summary, "blurbs": validated}
        return "Supervisor analysis complete.", artifact

    def get_merged_allowlist() -> dict[
        str, tuple[str, Literal["competitor", "mechanism", "both"]]
    ]:
        """Snapshot the post-merge competitor + mechanism disease allowlist.

        Returns a copy keyed by lowercase disease name → (canonical_name, source). Sources are
        "competitor", "mechanism", or "both" depending on which sub-agent surfaced the disease.
        """
        return dict(allowed_diseases)

    def get_auto_findings() -> dict[str, dict]:
        """Snapshot artifacts produced by investigate_top_candidates (holdout-only).

        Returns {lowercase_canonical_disease: {"literature": LiteratureOutput,
        "clinical_trials": ClinicalTrialsOutput}}. Empty in non-holdout runs.
        """
        return dict(auto_findings)

    tools = [
        find_candidates,
        analyze_mechanism,
        analyze_literature,
        analyze_clinical_trials,
        get_drug_briefing,
        finalize_supervisor,
    ]
    if date_before is not None:
        # Holdout-only: insert investigate_top_candidates before finalize so
        # the LLM can see it after seed-phase tools but before terminating.
        tools.insert(-1, investigate_top_candidates)
    return tools, get_merged_allowlist, get_auto_findings
