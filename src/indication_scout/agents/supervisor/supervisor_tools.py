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
from indication_scout.services.approval_check import get_fda_approved_diseases

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

    @tool(response_format="content_and_artifact")
    async def find_candidates(drug_name: str) -> tuple[str, list[str]]:
        """Surface candidate diseases for repurposing this drug.

        Uses Open Targets to find diseases where competitor drugs (drugs sharing the same molecular
        targets) are being developed. Returns a list of disease names ranked by competitor activity.
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

        Investigates published evidence via PubMed, embeds and re-ranks abstracts, and produces a
        structured evidence summary with strength rating (none / weak / moderate / strong).
        """
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

        Checks ClinicalTrials.gov for existing trials, competitive landscape, and terminated
        trials (safety/efficacy red flags).
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
        """Run the mechanism sub-agent for a drug.

        The mechanism agent returns target-level MoA data and the agent's narrative summary. No
        disease-level hypothesis surfacing in this step — rebuilt in a follow-up.
        """
        output = await run_mechanism_agent(mech_agent, drug_name)
        logger.warning("[TOOL] analyze_mechanism(drug=%r)", drug_name)

        n_targets = len(output.drug_targets)
        summary = (
            f"Mechanism analysis for {drug_name}: {n_targets} targets."
        )
        return summary, output

    @tool(response_format="content_and_artifact")
    async def finalize_supervisor(summary: str) -> tuple[str, str]:
        """Signal that the repurposing analysis is complete.

        Call this as the very last step, passing your 4-6 sentence plain-text summary of the most
        promising candidates. This terminates the agent loop.
        """
        return "Supervisor analysis complete.", summary

    return [find_candidates, analyze_mechanism, analyze_literature, analyze_clinical_trials, finalize_supervisor]
