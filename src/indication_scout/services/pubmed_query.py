from indication_scout.services.llm import query_small_llm


async def get_pubmed_query(drug_name, disease_name):
    """Construct a PubMed search query for a drug-disease pair.
       Uses an LLM to convert the disease name into the best PubMed search term.
       """
    prompt = f"""Convert this disease name to the best PubMed search query.
    Return ONLY the search term, nothing else.
    Disease: {disease_name}"""

    pubmed_term = await query_small_llm(prompt)
    return f"{drug_name} AND {pubmed_term}"
