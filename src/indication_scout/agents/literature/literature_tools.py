import json
import logging
from datetime import date

from langchain_core.tools import tool


from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


def build_literature_tools(svc: RetrievalService, drug_profile:DrugProfile, date_before: date | None = None, max_search_results:int=50) -> list:

    @tool
    async def expand_search_terms(drug_name: str, disease_name: str) -> list[str]:
        """Generate diverse PubMed keyword queries for a drug-disease pair.

        Uses the drug profile (synonyms, targets, MOA, ATC codes) to produce
        5-10 varied queries. Always call this first.
        """
        return await svc.expand_search_terms(drug_name, disease_name, drug_profile)


    return [expand_search_terms]
