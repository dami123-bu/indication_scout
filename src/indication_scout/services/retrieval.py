"""Retrieval service: PubMed fetch/embed/cache and semantic search via pgvector."""

from pathlib import Path

from indication_scout.services.llm import query_small_llm, parse_llm_response

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


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


async def expand_search_terms(
    drug_name: str, disease_name: str, drug_profile: dict
) -> list[str]:
    """Use LLM to generate PubMed search queries from drug-disease pair."""
    raise NotImplementedError


async def get_disease_synonyms(disease):
    template = (_PROMPTS_DIR / "disease_synonyms.txt").read_text()
    prompt = template.format(disease=disease)

    text = await query_small_llm(prompt)
    return parse_llm_response(text)
