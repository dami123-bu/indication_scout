import logging
from datetime import date

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


def build_literature_tools(
    svc: RetrievalService,
    db: Session,
    drug_profile: DrugProfile,
    date_before: date | None = None,
    max_search_results: int | None = None,
    num_top_k: int = 5,
) -> list:

    @tool(response_format="content_and_artifact")
    async def expand_search_terms(
        drug_name: str, disease_name: str
    ) -> tuple[str, list[str]]:
        """Generate diverse PubMed keyword queries for a drug-disease pair.

        Uses the drug profile (synonyms, targets, MOA, ATC codes) to produce
        5-10 varied queries. Always call this first.
        """
        result = await svc.expand_search_terms(drug_name, disease_name, drug_profile)
        return f"Generated {len(result)} search terms", result

    @tool(response_format="content_and_artifact")
    async def fetch_and_cache(search_terms: list[str]) -> tuple[str, list[str]]:
        """Search PubMed for all search terms, embed abstracts, and store in pgvector.

        Call this after expand_search_terms, passing the search terms it returned.
        Returns pmids.
        """
        pmids = await svc.fetch_and_cache(
            search_terms, db, date_before=date_before, max_results=max_search_results
        )
        return f"Fetched and cached {len(pmids)} abstracts. PMIDs: {pmids}", pmids

    @tool(response_format="content_and_artifact")
    async def semantic_search(
        drug_name: str, disease_name: str, pmids: list[str]
    ) -> tuple[str, list[dict]]:
        """Retrieve top-k abstracts from pgvector most similar to the drug-disease query.

        Restricted to the supplied PMIDs.
        """
        result = await svc.semantic_search(
            disease_name, drug_name, pmids, db, num_top_k
        )
        return f"Retrieved {len(result)} relevant abstracts: {result}", result

    @tool(response_format="content_and_artifact")
    async def synthesize(
        drug_name: str, disease_name: str, abstracts: list[dict]
    ) -> tuple[str, EvidenceSummary]:
        """Synthesize retrieved abstracts into a structured EvidenceSummary.

        Returns strength (strong/moderate/weak/none), study_count, study_types,
        key_findings, has_adverse_effects, and supporting_pmids.
        Call this once you have high-similarity abstracts. If no evidence was
        found, call with whatever is available — do not skip.
        """
        result = await svc.synthesize(drug_name, disease_name, abstracts)
        return (
            f"Synthesis complete: strength={result.strength}, study_count={result.study_count}",
            result,
        )

    return [expand_search_terms, fetch_and_cache, semantic_search, synthesize]
