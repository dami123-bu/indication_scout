import json
import logging
from datetime import date

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


def build_literature_tools(
    svc: RetrievalService,
    db:Session,
    drug_profile: DrugProfile,
    date_before: date | None = None,
    max_search_results: int | None = None,
) -> list:

    @tool
    async def expand_search_terms(drug_name: str, disease_name: str) -> list[str]:
        """Generate diverse PubMed keyword queries for a drug-disease pair.

        Uses the drug profile (synonyms, targets, MOA, ATC codes) to produce
        5-10 varied queries. Always call this first.
        """
        return await svc.expand_search_terms(drug_name, disease_name, drug_profile)

    @tool
    async def fetch_and_cache(search_terms: list[str]) -> dict:
        """Search PubMed for all search terms, embed abstracts, and store in pgvector.

        Call this after expand_search_terms, passing the search terms it returned.
        Returns pmid_count and pmids.
        """
        pmids = await svc.fetch_and_cache(search_terms, db, date_before=date_before, max_results=max_search_results)
        return {"pmid_count": len(pmids), "pmids": pmids}

    return [expand_search_terms, fetch_and_cache]
