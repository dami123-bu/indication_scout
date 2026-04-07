"""Mechanism tools.

Uses content_and_artifact so typed Python objects are preserved on
msg.artifact. Tools share inter-call data via a closure-scoped store dict.
No InjectedState, no LangGraph state machinery.
"""

from langchain_core.tools import tool

from indication_scout.data_sources.open_targets import CompetitorRawData, OpenTargetsClient
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.services.retrieval import RetrievalService


def build_mechanism_tools(svc: RetrievalService) -> list:
    """Build tools that share data via a closure-scoped store dict.

    Tools write to the store themselves as a side effect, so subsequent
    tools can read prior results without the LLM passing them around.
    """

    store: dict = {}

    @tool(response_format="content_and_artifact")
    async def build_drug_profile(drug_name: str) -> tuple[str, DrugProfile]:
        """Fetch pharmacological profile (synonyms, gene targets, mechanisms)
        for a drug. Call before expand_search_terms for richer queries."""
        profile = await svc.build_drug_profile(drug_name)
        store["drug_profile"] = profile
        return f"Profile for {drug_name}: {len(profile.synonyms)} synonyms", profile

    @tool(response_format="content_and_artifact")
    async def get_drug_competitors(drug_name: str) -> tuple[str, CompetitorRawData]:
        """Get competitor drugs for a drug, grouped by disease.

        Returns drugs sharing the same molecular targets, grouped by disease,
        and the drug's approved indications.
        """
        async with OpenTargetsClient() as client:
            competitors = await client.get_drug_competitors(drug_name)

        store["competitors"] = competitors
        total = sum(len(drugs) for drugs in competitors["diseases"].values())
        return f"Fetched {total} competitor drugs across {len(competitors['diseases'])} diseases", competitors

    @tool(response_format="content_and_artifact")
    async def expand_search_terms(drug_name: str, disease: str) -> tuple[str, list[str]]:
        """Generate PubMed search queries for a drug-disease pair.

        Reads the drug profile from the store if available (call build_drug_profile first).
        Call once per disease from the competitors dict.
        """
        profile = store.get("drug_profile") or await svc.build_drug_profile(drug_name)
        queries = await svc.expand_search_terms(drug_name, disease, profile)
        return f"Generated {len(queries)} queries for {disease}", queries

    return [
        get_drug_competitors,
        build_drug_profile,
        expand_search_terms,
    ]
