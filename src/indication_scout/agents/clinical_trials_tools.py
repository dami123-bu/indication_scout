from langchain_core.tools import tool
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient

async def get_terminated(query:str) -> list[dict]:
    async with ClinicalTrialsClient() as client:
        results = await client.get_terminated(query)
    return [t.model_dump for t in results [20:]]