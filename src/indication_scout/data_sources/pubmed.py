"""PubMed API client."""


class PubMedClient(BaseClient):
    """Client for querying PubMed/NCBI APIs."""

    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    @property
    def _source_name(self) -> str:
        return "pubmed"

    async def search(
        self, query: str, max_results: int = 50, date_before: date | None = None
    ) -> PartialResult:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        params = self.apply_date_filter(params, date_before, "datetype_maxdate")

        return await self._rest_get(
            self.SEARCH_URL,
            params,
            cache_namespace="pubmed_search",
            context=RequestContext(source=self._source_name, method="search"),
        )
