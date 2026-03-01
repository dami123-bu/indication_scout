"""Retrieval service: PubMed fetch/embed/cache and semantic search via pgvector."""

import logging
from pathlib import Path

from indication_scout.constants import CACHE_TTL, DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.services.llm import parse_llm_response, query_small_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def build_drug_profile(drug_name: str) -> DrugProfile:
    """Fetch drug + target data from Open Targets, enrich with ATC descriptions from ChEMBL,
    and return a DrugProfile ready for use in search term expansion.

    Args:
        drug_name: Common drug name (e.g. "metformin").

    Returns:
        DrugProfile with all fields populated. atc_descriptions will be [] if the drug
        has no ATC classifications.
    """
    async with OpenTargetsClient() as open_targets_client:
        rich = await open_targets_client.get_rich_drug_data(drug_name)

    atc_descriptions = []
    if rich.drug.atc_classifications:
        async with ChEMBLClient() as chembl_client:
            for code in rich.drug.atc_classifications:
                atc_descriptions.append(await chembl_client.get_atc_description(code))

    return DrugProfile.from_rich_drug_data(rich, atc_descriptions)


async def fetch_and_cache(queries: list[str]) -> list[str]:
    """Hit PubMed for each query, fetch new abstracts, embed, cache in pgvector. Return PMIDs."""
    raise NotImplementedError


async def semantic_search(disease: str, drug: str, top_k: int = 20) -> list[dict]:
    """Create a query from drug + disease, embed it, search pgvector, return ranked abstracts.
    [
        {"pmid": "29734553", "title": "Metformin suppresses colorectal...", "abstract": "...", "similarity": 0.89},
        {"pmid": "31245678", "title": "AMPK activation in colon...", "abstract": "...", "similarity": 0.85},
        {"pmid": "30198432", "title": "Biguanide compounds inhibit...", "abstract": "...", "similarity": 0.82},
    ...
    ]
    """
    raise NotImplementedError


def synthesize(drug, disease, top_5_abstracts):
    """get back a structured evidence summary with PMIDs
    Summarize the evidence for Metformin treating colorectal cancer based on these papers" and it returns a structured summary
     — study count, study types, strength assessment, key findings, and the PMIDs it drew from.
    That evidence summary is the final output of the RAG pipeline for one drug-disease pair.
    """
    raise NotImplementedError


async def extract_organ_term(disease_name: str) -> str:
    """Return the primary organ or tissue for a disease name via a small LLM call."""
    cached = cache_get("organ_term", {"disease_name": disease_name}, DEFAULT_CACHE_DIR)
    if cached is not None:
        logger.debug("Cache hit for organ_term: %s", disease_name)
        return cached

    template = (_PROMPTS_DIR / "extract_organ_term.txt").read_text()
    prompt = template.format(disease_name=disease_name)
    result = await query_small_llm(prompt)
    organ_term = result.strip()

    cache_set(
        "organ_term",
        {"disease_name": disease_name},
        organ_term,
        DEFAULT_CACHE_DIR,
        ttl=CACHE_TTL,
    )
    logger.debug("Extracted organ term '%s' for disease '%s'", organ_term, disease_name)
    return organ_term


async def expand_search_terms(
    drug_name: str, disease_name: str, drug_profile: DrugProfile
) -> list[str]:
    """Use LLM to generate diverse PubMed search queries from a drug-disease pair."""
    cached = cache_get(
        "expand_search_terms",
        {"drug_name": drug_name, "disease_name": disease_name},
        DEFAULT_CACHE_DIR,
    )
    if cached is not None:
        logger.debug(
            "Cache hit for expand_search_terms: %s / %s", drug_name, disease_name
        )
        return cached

    organ_term = await extract_organ_term(disease_name)

    template = (_PROMPTS_DIR / "expand_search_terms.txt").read_text()
    prompt = template.format(
        drug_name=drug_name,
        disease_name=disease_name,
        organ_term=organ_term,
        synonyms=", ".join(drug_profile.synonyms),
        target_gene_symbols=", ".join(drug_profile.target_gene_symbols),
        mechanisms_of_action=", ".join(drug_profile.mechanisms_of_action),
        atc_codes=", ".join(drug_profile.atc_codes),
        atc_descriptions=", ".join(drug_profile.atc_descriptions),
        drug_type=drug_profile.drug_type,
    )

    text = await query_small_llm(prompt)
    raw: list[str] = parse_llm_response(text)

    # Case-normalised dedup: lowercase+strip as key, preserve original casing
    seen: dict[str, str] = {}
    for term in raw:
        key = term.lower().strip()
        if key not in seen:
            seen[key] = term
    deduped = list(seen.values())

    cache_set(
        "expand_search_terms",
        {"drug_name": drug_name, "disease_name": disease_name},
        deduped,
        DEFAULT_CACHE_DIR,
        ttl=CACHE_TTL,
    )
    logger.debug(
        "expand_search_terms returned %d queries for %s / %s",
        len(deduped),
        drug_name,
        disease_name,
    )
    return deduped


async def get_disease_synonyms(disease):
    template = (_PROMPTS_DIR / "disease_synonyms.txt").read_text()
    prompt = template.format(disease=disease)

    text = await query_small_llm(prompt)
    return parse_llm_response(text)
