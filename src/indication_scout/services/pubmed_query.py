from pathlib import Path

from indication_scout.services.llm import parse_llm_response, query_small_llm

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def get_pubmed_query(drug_name, disease_name):
    """Construct a PubMed search query for a drug-disease pair.
    Uses an LLM to convert the disease name into the best PubMed search term.
    """
    prompt = f"""Convert this disease name to the best PubMed search query.
    Return ONLY the search term, nothing else.
    Disease: {disease_name}"""

    pubmed_term = await query_small_llm(prompt)
    return f"{drug_name} AND {pubmed_term}"


async def get_disease_synonyms(disease):
    template = (_PROMPTS_DIR / "disease_synonyms.txt").read_text()
    prompt = template.format(disease=disease)

    text = await query_small_llm(prompt)
    return parse_llm_response(text)
