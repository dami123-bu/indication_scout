"""Supervisor tools — wraps sub-agents as tools.

Each sub-agent (literature, clinical trials) becomes a single tool the supervisor can call. The
tool runs the full sub-agent and returns its typed output as an artifact, plus a short summary
string for the LLM.

There's also a find_candidates tool that hits Open Targets directly to surface disease candidates
for a drug.
"""

import logging
import time
from typing import Literal

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.data_sources.fda import FDAClient
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.services.approval_check import (
    get_fda_approved_disease_mapping,
    list_approved_indications_from_labels,
)

logger = logging.getLogger(__name__)

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
) -> list:
    """Build supervisor tools that close over the sub-agents.

    The literature and clinical trials agents are compiled once here and reused across calls — no
    need to rebuild them per invocation.
    """

    # Build sub-agents once at supervisor construction (except literature — see below)
    ct_agent = build_clinical_trials_agent(llm=llm)
    mech_agent = build_mechanism_agent(llm=llm)

    # Closure-scoped allowlist — populated by find_candidates and analyze_mechanism, checked by
    # analyze_literature / analyze_clinical_trials.
    # allowed_diseases: lowercase disease name → (canonical_name, source)
    allowed_diseases: dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]] = {}

    # Drug-level shared store. Populated by sub-agents as they run; surfaced
    # to the supervisor via get_drug_briefing. Keyed by drug name (normalized).
    # See supervisor_ideas.md for design rationale.
    drug_facts: dict[str, dict] = {}

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
        drug_name = normalize_drug_name(drug_name)
        chembl_id = await resolve_drug_name(drug_name, svc.cache_dir)
        competitors = await svc.get_drug_competitors(chembl_id)
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
        seed_aliases = entry["drug_aliases"] or [drug_name]
        try:
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
                logger.warning(
                    "[TOOL] find_candidates seeded %d approved indication(s) from label: %s",
                    len(seeded), seeded,
                )
        except Exception as e:
            logger.warning(
                "find_candidates: label-derived approved-indication seed failed for %s: %s",
                drug_name, e,
            )

        # FDA approval check — drop competitor diseases already approved for this drug
        fda_approved_lower: set[str] = set()
        if diseases:
            mapping = await get_fda_approved_disease_mapping(
                drug_name=drug_name,
                candidate_diseases=diseases,
                cache_dir=svc.cache_dir,
            )
            fda_approved = {disease for disease, is_approved in mapping.items() if is_approved}
            if fda_approved:
                logger.warning(
                    "[TOOL] find_candidates FDA approval check removing %d competitor diseases: %s",
                    len(fda_approved), fda_approved,
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

        allowed_diseases.clear()
        for d in diseases:
            allowed_diseases[d.lower().strip()] = (d, "competitor")
        logger.warning("[TOOL] find_candidates(%r [%s]) -> %s", drug_name, chembl_id, diseases)
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
        logger.warning("[TOOL] %s REJECTED disease=%r", tool_label, disease_name)
        return msg, empty_output

    @tool(response_format="content_and_artifact")
    async def analyze_literature(
        drug_name: str, disease_name: str
    ) -> tuple[str, LiteratureOutput]:
        """Run a full literature analysis for a drug-disease pair.

        Investigates published evidence via PubMed, embeds and re-ranks abstracts, and produces a
        structured evidence summary with strength rating (none / weak / moderate / strong).
        """
        drug_name = normalize_drug_name(drug_name)
        # Build a fresh agent per call so the closure-scoped store dict in literature_tools is not
        # shared across disease invocations.
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(disease_name, "analyze_literature", LiteratureOutput())

        logger.warning("[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name)


        lit_agent = build_literature_agent(llm=llm, svc=svc, db=db)
        t0 = time.perf_counter()
        logger.warning("[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name)

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
        drug_name = normalize_drug_name(drug_name)
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(disease_name, "analyze_clinical_trials", ClinicalTrialsOutput())

        logger.warning("[TOOL] analyze_clinical_trials(drug=%r, disease=%r)", drug_name, disease_name)
        t0 = time.perf_counter()
        output = await run_clinical_trials_agent(ct_agent, drug_name, disease_name)
        logger.info("analyze_clinical_trials took %.2fs for %s × %s", time.perf_counter() - t0, drug_name, disease_name)

        # Short-circuit cases: the clinical trials sub-agent stops calling
        # trial tools when the FDA check returns a definitive answer. In
        # those cases, search/completed/terminated are None — surface the
        # approval status to the supervisor instead of zeros.
        approval = output.approval
        if approval is not None and approval.is_approved:
            # Drug-level write-through: capture the approved indication so
            # the supervisor's briefing reflects what we discovered while
            # analyzing this candidate.
            entry = _ensure_drug_entry(drug_name)
            matched = approval.matched_indication or disease_name
            existing = {ind.lower().strip() for ind in entry["approved_indications"]}
            if matched.lower().strip() not in existing:
                entry["approved_indications"].append(matched)
            summary = (
                f"Clinical trials for {drug_name} × {disease_name}: "
                f"{drug_name} is FDA-approved for {disease_name} — not a "
                f"repurposing opportunity."
            )
            return summary, output
        if approval is not None and approval.label_found is False:
            summary = (
                f"Clinical trials for {drug_name} × {disease_name}: "
                f"no FDA label found for {drug_name} (drug may be withdrawn, "
                f"never approved, or approved outside the US); approval status "
                f"unknown, trial analysis skipped."
            )
            return summary, output

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
        sub_agent_summary = output.summary or ""
        summary = (
            f"{header}\n\n{sub_agent_summary}".strip()
            if sub_agent_summary
            else header
        )
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_mechanism(drug_name: str) -> tuple[str, MechanismOutput]:
        """Run the mechanism sub-agent for a drug.

        The mechanism agent returns target-level MoA data and the agent's narrative summary.
        Mechanism-surfaced candidates are promoted into the investigation allowlist so
        analyze_literature / analyze_clinical_trials can investigate them downstream.
        """
        drug_name = normalize_drug_name(drug_name)
        output = await run_mechanism_agent(mech_agent, drug_name)
        logger.warning("[TOOL] analyze_mechanism(drug=%r)", drug_name)

        promoted: list[str] = []
        for candidate in output.candidates:
            key = candidate.disease_name.lower().strip()
            if not key:
                continue
            if key in allowed_diseases:
                existing_name, source = allowed_diseases[key]
                if source == "competitor":
                    allowed_diseases[key] = (existing_name, "both")
            else:
                allowed_diseases[key] = (candidate.disease_name, "mechanism")
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

    @tool(response_format="content_and_artifact")
    async def finalize_supervisor(summary: str) -> tuple[str, str]:
        """Signal that the repurposing analysis is complete.

        Call this as the very last step, passing your 4-6 sentence plain-text summary of the most
        promising candidates. This terminates the agent loop.
        """
        return "Supervisor analysis complete.", summary

    return [
        find_candidates,
        analyze_mechanism,
        analyze_literature,
        analyze_clinical_trials,
        get_drug_briefing,
        finalize_supervisor,
    ]
