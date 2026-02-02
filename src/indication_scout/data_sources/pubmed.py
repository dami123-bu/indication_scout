"""PubMed API client."""


class PubMedClient:
    """Client for querying PubMed/NCBI APIs."""

    async def search(self, query: str, max_results: int = 100) -> list[dict]:
        """Search PubMed for articles matching query."""
        raise NotImplementedError

    async def fetch_abstract(self, pmid: str) -> str:
        """Fetch abstract for a given PubMed ID."""
        raise NotImplementedError
