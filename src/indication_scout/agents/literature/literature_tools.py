"""LangChain tools wrapping RetrievalService for the literature agent.

Each tool uses response_format="content_and_artifact" so the LLM sees a
short string summary while the typed Python object rides on msg.artifact
straight into state — no JSON round-trip.

Tools that depend on earlier results (e.g. semantic_search needs pmids)
read from a shared store dict that tools_node updates after each round.
This avoids InjectedState entirely — pure closures, no magic.
"""

from datetime import date

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult, RetrievalService


def build_literature_tools(
    svc: RetrievalService,
    db: Session,
    date_before: date | None = None,
    max_search_results: int | None = None,
    num_top_k: int = 5,
) -> tuple[list, dict]:
    """Build tools that close over the retrieval service, DB session, and a
    shared store for inter-tool data flow.

    Returns:
        (tools, store) — caller must update store after each tool round
        so subsequent tools can read prior results.
    """

    # Mutable dict shared across all tools via closure.
    # tools_node writes to it after each round; tools read from it.
    store: dict = {}

    @tool(response_format="content_and_artifact")
    async def build_drug_profile(drug_name: str) -> tuple[str, DrugProfile]:
        """Fetch pharmacological profile for a drug from Open Targets and ChEMBL.

        Returns synonyms, gene targets, mechanisms of action, and ATC codes.
        Call this before expand_search_terms to get mechanism-aware queries.
        """
        profile = await svc.build_drug_profile(drug_name)
        summary = (
            f"Profile for {drug_name}: "
            f"{len(profile.synonyms)} synonyms, "
            f"{len(profile.target_gene_symbols)} gene targets, "
            f"{len(profile.mechanisms_of_action)} mechanisms"
        )
        return summary, profile

    @tool(response_format="content_and_artifact")
    async def expand_search_terms(
        drug_name: str,
        disease_name: str,
    ) -> tuple[str, list[str]]:
        """Generate diverse PubMed keyword queries for a drug-disease pair.

        Uses the drug profile (from build_drug_profile) to create queries
        covering synonyms, gene targets, and mechanisms. If no drug profile
        is available yet, builds one automatically.
        """
        profile = store.get("drug_profile")
        if profile is None:
            profile = await svc.build_drug_profile(drug_name)

        queries = await svc.expand_search_terms(drug_name, disease_name, profile)
        summary = (
            f"Generated {len(queries)} search queries for {drug_name} × {disease_name}"
        )
        return summary, queries

    @tool(response_format="content_and_artifact")
    async def fetch_and_cache(
        drug_name: str,
    ) -> tuple[str, list[str]]:
        """Search PubMed with the expanded queries, fetch abstracts, embed
        with BioLORD-2023, and cache in pgvector.

        Reads search queries from prior tool results.
        Returns deduplicated PMIDs across all queries.
        """
        queries = store.get("expanded_search_results", [])
        if not queries:
            return "No search queries available — call expand_search_terms first.", []

        pmids = await svc.fetch_and_cache(
            queries, db, date_before=date_before, max_results=max_search_results
        )
        summary = f"Fetched and cached {len(pmids)} PMIDs from {len(queries)} queries"
        return summary, pmids

    @tool(response_format="content_and_artifact")
    async def semantic_search(
        drug_name: str,
        disease_name: str,
    ) -> tuple[str, list[AbstractResult]]:
        """Re-rank cached abstracts by semantic similarity to the drug-disease pair.

        Reads PMIDs from prior tool results. Returns the top-k most relevant
        abstracts with similarity scores.
        """
        pmids = store.get("pmids", [])
        if not pmids:
            return "No PMIDs available — call fetch_and_cache first.", []

        results = await svc.semantic_search(
            disease_name, drug_name, pmids, db, top_k=num_top_k
        )

        if not results:
            summary = f"No relevant abstracts found for {drug_name} × {disease_name}"
        else:
            top_sim = results[0].similarity
            summary = (
                f"Found {len(results)} relevant abstracts "
                f"(top similarity: {top_sim:.2f}) for {drug_name} × {disease_name}"
            )
        return summary, results

    @tool(response_format="content_and_artifact")
    async def synthesize(
        drug_name: str,
        disease_name: str,
    ) -> tuple[str, EvidenceSummary]:
        """Synthesize retrieved abstracts into a structured evidence summary.

        Reads abstracts from prior tool results. Uses an LLM to assess
        evidence strength, key findings, and gaps. Call this even when
        evidence is weak — negative findings are valuable.
        """
        abstracts = store.get("semantic_search_results", [])

        evidence = await svc.synthesize(drug_name, disease_name, abstracts)
        summary = (
            f"Evidence for {drug_name} × {disease_name}: "
            f"strength={evidence.strength}"
        )
        return summary, evidence

    tools = [
        build_drug_profile,
        expand_search_terms,
        fetch_and_cache,
        semantic_search,
        synthesize,
    ]

    return tools, store
