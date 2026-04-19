"""Supervisor tools — wraps sub-agents as tools.

Each sub-agent (literature, clinical trials) becomes a single tool the
supervisor can call. The tool runs the full sub-agent and returns its
typed output as an artifact, plus a short summary string for the LLM.

There's also a find_candidates tool that hits Open Targets directly to
surface disease candidates for a drug.
"""

import logging
import time
from collections import Counter
from typing import Literal

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.constants import MECHANISM_ASSOCIATION_MIN_SCORE
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.services.approval_check import get_fda_approved_diseases, remove_approved_from_labels

logger = logging.getLogger(__name__)

from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
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

    The literature and clinical trials agents are compiled once here and
    reused across calls — no need to rebuild them per invocation.
    """

    # Build sub-agents once at supervisor construction (except literature — see below)
    ct_agent = build_clinical_trials_agent(llm=llm)
    mech_agent = build_mechanism_agent(llm=llm)

    # Closure-scoped allowlist — populated by find_candidates and
    # analyze_mechanism, checked by analyze_literature / analyze_clinical_trials.
    # allowed_diseases: lowercase disease name → (canonical_name, source)
    allowed_diseases: dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]] = {}

    @tool(response_format="content_and_artifact")
    async def find_candidates(drug_name: str) -> tuple[str, list[str]]:
        """Surface candidate diseases for repurposing this drug.

        Uses Open Targets to find diseases where competitor drugs (drugs
        sharing the same molecular targets) are being developed. Returns
        a list of disease names ranked by competitor activity.
        """
        chembl_id = await resolve_drug_name(drug_name, svc.cache_dir)
        competitors = await svc.get_drug_competitors(chembl_id)
        diseases = list(competitors.keys())

        # FDA approval check — drop competitor diseases already approved for this drug
        fda_approved_lower: set[str] = set()
        if diseases:
            drug_names = await get_all_drug_names(chembl_id, svc.cache_dir)
            if drug_names:
                fda_approved = await get_fda_approved_diseases(
                    drug_names=drug_names,
                    candidate_diseases=diseases,
                    cache_dir=svc.cache_dir,
                )
                if fda_approved:
                    logger.warning(
                        "[TOOL] find_candidates FDA approval check removing %d competitor diseases: %s",
                        len(fda_approved), fda_approved,
                    )
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

        Investigates published evidence via PubMed, embeds and re-ranks
        abstracts, and produces a structured evidence summary with strength
        rating (none / weak / moderate / strong).
        """
        # Build a fresh agent per call so the closure-scoped store dict in
        # literature_tools is not shared across disease invocations.
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
        summary = (
            f"Literature for {drug_name} × {disease_name}: "
            f"{len(output.pmids)} PMIDs, strength={strength}"
        )
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_clinical_trials(
        drug_name: str, disease_name: str
    ) -> tuple[str, ClinicalTrialsOutput]:
        """Run a full clinical trials analysis for a drug-disease pair.

        Checks ClinicalTrials.gov for existing trials, competitive landscape,
        and terminated trials (safety/efficacy red flags).
        """
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(disease_name, "analyze_clinical_trials", ClinicalTrialsOutput())

        logger.warning("[TOOL] analyze_clinical_trials(drug=%r, disease=%r)", drug_name, disease_name)
        t0 = time.perf_counter()
        output = await run_clinical_trials_agent(ct_agent, drug_name, disease_name)
        logger.info("analyze_clinical_trials took %.2fs for %s × %s", time.perf_counter() - t0, drug_name, disease_name)

        active_statuses = {
            "RECRUITING",
            "ACTIVE_NOT_RECRUITING",
            "ENROLLING_BY_INVITATION",
            "NOT_YET_RECRUITING",
        }
        stopped_statuses = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
        n_active = 0
        n_completed = 0
        n_stopped = 0
        for t in output.trials:
            status = (t.overall_status or "").upper()
            if status in active_statuses:
                n_active += 1
            elif status == "COMPLETED":
                n_completed += 1
            elif status in stopped_statuses:
                n_stopped += 1
        pair_safety_efficacy = sum(
            1 for t in output.terminated.pair_specific
            if t.stop_category in {"safety", "efficacy"}
        )
        pair_completed_phase3 = sum(
            1 for t in output.terminated.pair_completed
            if "3" in (t.phase or "")
        )
        summary = (
            f"Clinical trials for {drug_name} × {disease_name}: "
            f"{n_active} active, {n_completed} completed, "
            f"{n_stopped} terminated/withdrawn/suspended. "
            f"Outcome evidence — "
            f"{len(output.terminated.drug_wide)} {drug_name} trials terminated "
            f"for safety/efficacy across all indications; "
            f"{len(output.terminated.indication_wide)} trials terminated in "
            f"{disease_name} historically (any drug); "
            f"{len(output.terminated.pair_specific)} {drug_name} trials in "
            f"{disease_name} stopped early "
            f"({pair_safety_efficacy} for safety/efficacy); "
            f"{len(output.terminated.pair_completed)} {drug_name} trials in "
            f"{disease_name} ran to completion "
            f"({pair_completed_phase3} of those Phase 3)."
        )
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_mechanism(drug_name: str) -> tuple[str, MechanismOutput]:
        """Run the mechanism sub-agent and filter out already-approved indications.

        Runs the mechanism sub-agent, collects target-disease associations that meet
        the score threshold, then removes any candidates already covered by
        the drug's FDA label (synonyms and narrower subsets included). What
        remains are live mechanism-sourced repurposing candidates, which are
        promoted into the investigation allowlist.
        """
        output = await run_mechanism_agent(mech_agent, drug_name)
        logger.warning("[TOOL] analyze_mechanism(drug=%r)", drug_name)

        mechanism_candidates: list[str] = []
        for assoc_list in output.associations.values():
            for assoc in assoc_list:
                if (assoc.overall_score or 0) >= MECHANISM_ASSOCIATION_MIN_SCORE:
                    key = assoc.disease_name.lower().strip()
                    if key not in allowed_diseases:
                        mechanism_candidates.append(assoc.disease_name)

        survivors: set[str] = set()
        if mechanism_candidates:
            chembl_id = await resolve_drug_name(drug_name, svc.cache_dir)
            drug_names = await get_all_drug_names(chembl_id, svc.cache_dir)
            if drug_names:
                survivors = await remove_approved_from_labels(
                    drug_names=drug_names,
                    candidate_diseases=mechanism_candidates,
                    cache_dir=svc.cache_dir,
                )
                dropped = set(mechanism_candidates) - survivors
                if dropped:
                    logger.warning(
                        "[TOOL] analyze_mechanism dropped %d mechanism diseases "
                        "(approval-covered or deduped): %s",
                        len(dropped), sorted(dropped),
                    )
            else:
                survivors = set(mechanism_candidates)

        survivors_lower = {s.lower().strip() for s in survivors}

        def _keep(assoc_disease_name: str, assoc_score: float | None) -> bool:
            """Keep below-threshold associations untouched; above-threshold must have survived."""
            if (assoc_score or 0) < MECHANISM_ASSOCIATION_MIN_SCORE:
                return True
            return assoc_disease_name.lower().strip() in survivors_lower

        original_association_count = sum(len(v) for v in output.associations.values())

        output.associations = {
            target: [a for a in assoc_list if _keep(a.disease_name, a.overall_score)]
            for target, assoc_list in output.associations.items()
        }
        # No overall_score on ShapedAssociation today — scrub by name only against
        # the above-threshold drop-set (mechanism_candidates minus survivors).
        dropped_lower = {c.lower().strip() for c in mechanism_candidates} - survivors_lower
        output.shaped_associations = [
            s for s in output.shaped_associations
            if s.disease_name.lower().strip() not in dropped_lower
        ]

        new_association_count = sum(len(v) for v in output.associations.values())
        scrubbed = original_association_count != new_association_count
        if scrubbed:
            # The LLM narrative was written pre-filter and may reference approved
            # indications. Blank it — the structured fields are the source of truth.
            output.summary = ""

        # Promote surviving mechanism candidates into the investigation allowlist.
        mechanism_added = 0
        for assoc_list in output.associations.values():
            for assoc in assoc_list:
                if (assoc.overall_score or 0) < MECHANISM_ASSOCIATION_MIN_SCORE:
                    continue
                key = assoc.disease_name.lower().strip()
                if key in allowed_diseases:
                    name, source = allowed_diseases[key]
                    if source == "competitor":
                        allowed_diseases[key] = (name, "both")
                else:
                    allowed_diseases[key] = (assoc.disease_name, "mechanism")
                    mechanism_added += 1

        logger.info(
            "analyze_mechanism added %d mechanism-sourced diseases to allowlist",
            mechanism_added,
        )

        mechanism_disease_names = sorted(
            canonical_name
            for _, (canonical_name, source) in allowed_diseases.items()
            if source in ("mechanism", "both")
        )

        n_targets = len(output.drug_targets)
        n_top = sum(len(a) for a in output.associations.values())
        shape_counts = Counter(s.shape for s in output.shaped_associations)
        shaped_lines = "\n".join(
            f"  [{s.shape.upper()}] {s.target_symbol} / {s.disease_name} ({s.disease_id}): {s.rationale}"
            for s in output.shaped_associations
        )

        mechanism_note = ""
        if mechanism_added > 0:
            mechanism_note = (
                f"\n{mechanism_added} mechanism-sourced diseases added to the "
                f"investigation allowlist (score >= {MECHANISM_ASSOCIATION_MIN_SCORE}): "
                f"{mechanism_disease_names}. You may now call analyze_literature or "
                f"analyze_clinical_trials with these disease names."
            )

        summary = (
            f"Mechanism analysis for {drug_name}: {n_targets} targets, "
            f"top {n_top} disease associations shown (capped per target).\n"
            f"Shaped associations: {dict(shape_counts)}\n"
            f"{shaped_lines}"
            f"{mechanism_note}"
        )
        return summary, output


    # @tool(response_format="content_and_artifact")
    # async def analyze_mechanism(drug_name: str) -> tuple[str, MechanismOutput]:
    #     """Analyse the molecular targets of a drug to provide mechanistic context.
    #
    #     Fetches the drug's targets, their top disease associations (with evidence
    #     scores by type), and their Reactome pathways. Drug-level — call once per
    #     drug, not once per candidate disease.
    #     """
    #     t0 = time.perf_counter()
    #     output = await run_mechanism_agent(mech_agent, drug_name)
    #     logger.warning("[TOOL] analyze_mechanism(drug=%r)", drug_name)
    #     logger.info("analyze_mechanism took %.2fs for %s", time.perf_counter() - t0, drug_name)
    #
    #     # Collect qualifying mechanism disease names not already in the allowlist
    #     mechanism_candidates: list[str] = []
    #     for assoc_list in output.associations.values():
    #         for assoc in assoc_list:
    #             if (assoc.overall_score or 0) >= MECHANISM_ASSOCIATION_MIN_SCORE:
    #                 key = assoc.disease_name.lower().strip()
    #                 if key not in allowed_diseases:
    #                     mechanism_candidates.append(assoc.disease_name)
    #
    #     # FDA approval check — remove already-approved diseases
    #     fda_approved_lower: set[str] = set()
    #     if mechanism_candidates:
    #         chembl_id = await resolve_drug_name(drug_name, svc.cache_dir)
    #         drug_names = await get_all_drug_names(chembl_id, svc.cache_dir)
    #         if drug_names:
    #             fda_approved = await get_fda_approved_diseases(
    #                 drug_names=drug_names,
    #                 candidate_diseases=mechanism_candidates,
    #                 cache_dir=svc.cache_dir,
    #             )
    #             if fda_approved:
    #                 logger.warning(
    #                     "[TOOL] FDA approval check removing %d mechanism diseases: %s",
    #                     len(fda_approved), fda_approved,
    #                 )
    #             fda_approved_lower = {d.lower().strip() for d in fda_approved}
    #
    #     # Drop FDA-approved diseases from the mechanism artifact so they do not
    #     # leak into the final SupervisorOutput.mechanism payload.
    #     if fda_approved_lower:
    #         output.associations = {
    #             target: [
    #                 a for a in assoc_list
    #                 if a.disease_name.lower().strip() not in fda_approved_lower
    #             ]
    #             for target, assoc_list in output.associations.items()
    #         }
    #         output.shaped_associations = [
    #             s for s in output.shaped_associations
    #             if s.disease_name.lower().strip() not in fda_approved_lower
    #         ]
    #         # Blank the LLM-generated narrative — it was written pre-filter and
    #         # may reference approved indications. Structured fields above are
    #         # the source of truth.
    #         output.summary = ""
    #
    #     # Promote mechanism associations above the score threshold into the
    #     # investigation allowlist so the LLM can run literature / trials on them.
    #     mechanism_added = 0
    #     for assoc_list in output.associations.values():
    #         for assoc in assoc_list:
    #             if (assoc.overall_score or 0) >= MECHANISM_ASSOCIATION_MIN_SCORE:
    #                 key = assoc.disease_name.lower().strip()
    #                 if key in fda_approved_lower:
    #                     continue  # FDA-approved — skip
    #                 if key in allowed_diseases:
    #                     # Already a competitor candidate — upgrade source to "both"
    #                     name, source = allowed_diseases[key]
    #                     if source == "competitor":
    #                         allowed_diseases[key] = (name, "both")
    #                 else:
    #                     allowed_diseases[key] = (assoc.disease_name, "mechanism")
    #                     mechanism_added += 1
    #     logger.info(
    #         "analyze_mechanism added %d mechanism-sourced diseases to allowlist",
    #         mechanism_added,
    #     )
    #
    #     # Build the list of mechanism-sourced diseases now in the allowlist
    #     # so the LLM knows it can investigate them.
    #     mechanism_disease_names = sorted(
    #         canonical_name
    #         for _, (canonical_name, source) in allowed_diseases.items()
    #         if source in ("mechanism", "both")
    #     )
    #
    #     n_targets = len(output.drug_targets)
    #     n_top = sum(len(a) for a in output.associations.values())
    #     shape_counts = Counter(s.shape for s in output.shaped_associations)
    #     shaped_lines = "\n".join(
    #         f"  [{s.shape.upper()}] {s.target_symbol} / {s.disease_name} ({s.disease_id}): {s.rationale}"
    #         for s in output.shaped_associations
    #     )
    #
    #     mechanism_note = ""
    #     if mechanism_added > 0:
    #         mechanism_note = (
    #             f"\n{mechanism_added} mechanism-sourced diseases added to the "
    #             f"investigation allowlist (score >= {MECHANISM_ASSOCIATION_MIN_SCORE}): "
    #             f"{mechanism_disease_names}. You may now call analyze_literature or "
    #             f"analyze_clinical_trials with these disease names."
    #         )
    #
    #     summary = (
    #         f"Mechanism analysis for {drug_name}: {n_targets} targets, "
    #         f"top {n_top} disease associations shown (capped per target).\n"
    #         f"Shaped associations: {dict(shape_counts)}\n"
    #         f"{shaped_lines}"
    #         f"{mechanism_note}"
    #     )
    #     return summary, output

    @tool(response_format="content_and_artifact")
    async def finalize_supervisor(summary: str) -> tuple[str, str]:
        """Signal that the repurposing analysis is complete.

        Call this as the very last step, passing your 4-6 sentence plain-text
        summary of the most promising candidates. This terminates the agent loop.
        """
        return "Supervisor analysis complete.", summary

    return [find_candidates, analyze_mechanism, analyze_literature, analyze_clinical_trials, finalize_supervisor]
