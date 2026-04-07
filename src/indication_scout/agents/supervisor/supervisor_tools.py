"""Supervisor tools — wraps sub-agents as tools.

Each sub-agent (literature, clinical trials) becomes a single tool the
supervisor can call. The tool runs the full sub-agent and returns its
typed output as an artifact, plus a short summary string for the LLM.

There's also a find_candidates tool that hits Open Targets directly to
surface disease candidates for a drug.
"""

from langchain_core.tools import tool
from sqlalchemy.orm import Session

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

    # Build sub-agents once at supervisor construction
    lit_agent = build_literature_agent(llm=llm, svc=svc, db=db)
    ct_agent = build_clinical_trials_agent(llm=llm)

    @tool(response_format="content_and_artifact")
    async def find_candidates(drug_name: str) -> tuple[str, list[str]]:
        """Surface candidate diseases for repurposing this drug.

        Uses Open Targets to find diseases where competitor drugs (drugs
        sharing the same molecular targets) are being developed. Returns
        a list of disease names ranked by competitor activity.
        """
        competitors = await svc.get_drug_competitors(drug_name)
        diseases = list(competitors.keys())
        return f"Found {len(diseases)} candidate diseases for {drug_name}", diseases

    @tool(response_format="content_and_artifact")
    async def analyze_literature(
        drug_name: str, disease_name: str
    ) -> tuple[str, LiteratureOutput]:
        """Run a full literature analysis for a drug-disease pair.

        Investigates published evidence via PubMed, embeds and re-ranks
        abstracts, and produces a structured evidence summary with strength
        rating (none / weak / moderate / strong).
        """
        output = await run_literature_agent(lit_agent, drug_name, disease_name)
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
        output = await run_clinical_trials_agent(ct_agent, drug_name, disease_name)
        n_trials = len(output.trials)
        n_terminated = len(output.terminated)
        summary = (
            f"Clinical trials for {drug_name} × {disease_name}: "
            f"{n_trials} active/completed, {n_terminated} terminated"
        )
        return summary, output

    return [find_candidates, analyze_literature, analyze_clinical_trials]
