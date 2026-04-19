import logging
from datetime import date

from langchain_core.tools import tool
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.models.model_clinical_trials import (
    WhitespaceResult,
    IndicationLandscape,
    Trial,
    TrialOutcomes,
)
from indication_scout.services.disease_helper import resolve_mesh_id

logger = logging.getLogger(__name__)


def build_clinical_trials_tools(
    date_before: date | None = None,
) -> list:

    @tool(response_format="content_and_artifact")
    async def get_terminated(
        drug: str, indication: str
    ) -> tuple[str, TrialOutcomes]:
        """Get trial-outcome evidence for a drug and indication, split by scope.

        Runs four queries and returns them in a TrialOutcomes with four
        scope-labelled lists:
        (1) drug_wide — this drug, ANY indication, safety/efficacy stop_category
            only. Reflects the drug's overall failure history.
        (2) indication_wide — this indication, ANY drug. Shows historical
            attrition in the disease area.
        (3) pair_specific — this drug AND this indication, TERMINATED. All
            stop_categories retained. Safety/efficacy here means the exact
            hypothesis has been directly tested and stopped early.
        (4) pair_completed — this drug AND this indication, COMPLETED.
            Catches trials that ran to protocol end (a Phase 3 that finishes
            but misses its primary endpoint is COMPLETED, not TERMINATED).
            Inspect these for likely outcome — a completed Phase 3 in a
            disease area without subsequent regulatory progression is a
            strong signal the endpoint was missed.

        When reporting, keep the scopes separate — do not sum them.
        """
        mesh_id = await resolve_mesh_id(indication)
        if mesh_id is None:
            logger.warning(
                "get_terminated: could not resolve MeSH id for indication '%s'; "
                "returning empty TrialOutcomes",
                indication,
            )
            return (
                f"Trial outcomes for {drug} × {indication}: MeSH unresolved, skipped.",
                TrialOutcomes(),
            )

        async with ClinicalTrialsClient() as client:
            outcomes = await client.get_terminated(
                drug,
                indication,
                date_before=date_before,
                sort="EnrollmentCount:desc",
                target_mesh_id=mesh_id,
            )

        pair_safety_efficacy = [
            t for t in outcomes.pair_specific
            if t.stop_category in {"safety", "efficacy"}
        ]
        pair_completed_phase3 = [
            t for t in outcomes.pair_completed
            if "3" in (t.phase or "")
        ]
        content = (
            f"Trial outcomes for {drug} × {indication}: "
            f"{len(outcomes.drug_wide)} drug-wide safety/efficacy failures "
            f"(across all indications), "
            f"{len(outcomes.indication_wide)} indication-specific terminations "
            f"(any drug in this space), "
            f"{len(outcomes.pair_specific)} pair-specific terminations "
            f"(this drug in this indication; "
            f"{len(pair_safety_efficacy)} safety/efficacy), "
            f"{len(outcomes.pair_completed)} pair-specific completed trials "
            f"({len(pair_completed_phase3)} Phase 3)"
        )
        return content, outcomes

    @tool(response_format="content_and_artifact")
    async def search_trials(drug: str, indication: str) -> tuple[str, list[Trial]]:
        """Search for clinical trials matching a drug and indication.

        Returns trial records including phase, status, enrollment, sponsor,
        interventions, and outcomes. Use when detect_whitespace shows trials
        exist and you need details.

        Only trials with a start date before the session cutoff are returned.
        """
        mesh_id = await resolve_mesh_id(indication)
        if mesh_id is None:
            logger.warning(
                "search_trials: could not resolve MeSH id for indication '%s'; "
                "returning empty list",
                indication,
            )
            return (
                f"Searched on {drug}-{indication}: MeSH unresolved, skipped.",
                [],
            )

        async with ClinicalTrialsClient() as client:
            trials = await client.search_trials(
                drug,
                indication,
                date_before=date_before,
                sort="EnrollmentCount:desc",
                target_mesh_id=mesh_id,
            )

        return f"Searched on {drug}-{indication} and found {len(trials)} trials", trials

    @tool(response_format="content_and_artifact")
    async def detect_whitespace(
        drug: str, indication: str
    ) -> tuple[str, WhitespaceResult]:
        """Check if a drug-indication pair has been explored in clinical trials.
        Returns whitespace signal: whether trials exist for this exact pair,
        how many trials exist for the drug alone and indication alone, and
        (when whitespace exists) other drugs being tested for the same indication.
        This should almost always be the first tool called.
        """
        mesh_id = await resolve_mesh_id(indication)
        if mesh_id is None:
            logger.warning(
                "detect_whitespace: could not resolve MeSH id for indication '%s'; "
                "returning empty WhitespaceResult",
                indication,
            )
            return (
                f"Whitespace check for {drug} × {indication}: MeSH unresolved, skipped.",
                WhitespaceResult(),
            )

        async with ClinicalTrialsClient() as client:
            whitespace = await client.detect_whitespace(
                drug,
                indication,
                date_before=date_before,
                target_mesh_id=mesh_id,
            )

        return (
            f"Whitespace check for {drug} × {indication}: {whitespace.is_whitespace}",
            whitespace,
        )

    @tool(response_format="content_and_artifact")
    async def get_landscape(indication: str) -> tuple[str, IndicationLandscape]:
        """Get the competitive landscape for a indication.

        Returns top 10 competitors grouped by sponsor + drug, ranked by
        phase then enrollment, plus phase distribution and recent starts.
        Use to understand how crowded the space is.
        """
        mesh_id = await resolve_mesh_id(indication)
        if mesh_id is None:
            logger.warning(
                "get_landscape: could not resolve MeSH id for indication '%s'; "
                "returning empty IndicationLandscape",
                indication,
            )
            return (
                f"Landscape for {indication}: MeSH unresolved, skipped.",
                IndicationLandscape(),
            )

        async with ClinicalTrialsClient() as client:
            landscape = await client.get_landscape(
                indication,
                date_before=date_before,
                top_n=10,
                target_mesh_id=mesh_id,
            )
        return (
            f"Landscape for {indication}: {len(landscape.competitors)} competitors",
            landscape,
        )

    @tool(response_format="content_and_artifact")
    async def finalize_analysis(summary: str) -> tuple[str, str]:
        """Signal that the analysis is complete.

        Call this as the very last step, passing your 2-3 sentence plain-text
        summary of the findings. This terminates the agent loop.
        """
        return "Analysis complete.", summary

    return [detect_whitespace, search_trials, get_landscape, get_terminated, finalize_analysis]
