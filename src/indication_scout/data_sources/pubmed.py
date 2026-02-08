"""
PubMed client for fetching publications via NCBI E-utilities.

Provides keyword search (get_key_publications) and direct fetch (fetch_by_pmids).
"""

import logging
import xml.etree.ElementTree as ET
from datetime import date

from indication_scout.data_sources.base_client import (
    BaseClient,
    ClientConfig,
    DataSourceError,
    RateLimitConfig,
    RequestContext,
)
from indication_scout.models.model_pubmed import Publication
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class PubMedClient(BaseClient):
    """
    Client for NCBI E-utilities (ESearch + EFetch).

    Rate limits:
    - Without API key: 3 requests/second
    - With API key: 10 requests/second
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    SEARCH_CACHE_TTL = 86400  # 24 hours
    FETCH_CACHE_TTL = 604800  # 7 days (published articles don't change)
    FETCH_BATCH_SIZE = 200  # max PMIDs per EFetch request

    def __init__(
        self,
        api_key: str | None = None,
        email: str = "contact@example.com",
        tool: str = "oncomind",
        config: ClientConfig | None = None,
    ):
        # Set rate limit based on API key presence
        rate = 10.0 if api_key else 3.0
        if config is None:
            config = ClientConfig(
                rate_limit=RateLimitConfig(requests_per_second=rate, burst=int(rate)),
            )

        super().__init__(config)

        self._api_key = api_key
        self._email = email
        self._tool = tool

    @property
    def _source_name(self) -> str:
        return "pubmed"

    def _base_params(self) -> dict[str, str]:
        """Return base parameters required on every NCBI request."""
        params = {
            "tool": self._tool,
            "email": self._email,
        }
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def get_key_publications(
        self,
        drug: str | None = None,
        condition: str | None = None,
        date_before: date | None = None,
        max_results: int = 20,
    ) -> list[Publication]:
        """
        Keyword search for publications matching a drug-condition query.

        Two-step process:
        1. ESearch - find PMIDs sorted by relevance (uses history server)
        2. EFetch - retrieve full XML records

        Parameters
        ----------
        drug : str, optional
            Drug name to search for in title/abstract.
        condition : str, optional
            Condition/disease to search for in title/abstract.
        date_before : date, optional
            Only return publications before this date (for temporal holdout).
        max_results : int
            Maximum number of publications to return (default 20).

        Returns
        -------
        list[Publication]
            Publications matching the query, sorted by relevance.

        Raises
        ------
        ValueError
            If neither drug nor condition is provided.
        """
        raise NotImplementedError("get_key_publications not yet implemented")

    async def fetch_by_pmids(
        self,
        pmids: list[str],
    ) -> list[Publication]:
        """
        Fetch full publication records for a list of known PMIDs.

        PMIDs may come from:
        - ClinicalTrials.gov trial references
        - Open Targets evidence links
        - Cross-references in other abstracts
        - Semantic search results

        Parameters
        ----------
        pmids : list[str]
            List of PubMed identifiers to fetch.

        Returns
        -------
        list[Publication]
            Publications for the requested PMIDs.
            Order may not match input order.
            Missing PMIDs are silently skipped.

        Notes
        -----
        - Empty input returns empty list (no API call)
        - Duplicate PMIDs are deduplicated before calling API
        - Batches requests in chunks of 200 PMIDs
        """
        if not pmids:
            return []

        # Deduplicate PMIDs
        unique_pmids = list(dict.fromkeys(pmids))

        all_publications: list[Publication] = []

        # Batch into chunks of FETCH_BATCH_SIZE
        for i in range(0, len(unique_pmids), self.FETCH_BATCH_SIZE):
            batch = unique_pmids[i : i + self.FETCH_BATCH_SIZE]
            publications = await self._fetch_batch(batch)
            all_publications.extend(publications)

        return all_publications

    async def _fetch_batch(self, pmids: list[str]) -> list[Publication]:
        """
        Fetch a single batch of PMIDs (up to FETCH_BATCH_SIZE).

        Parameters
        ----------
        pmids : list[str]
            List of PMIDs to fetch (max 200).

        Returns
        -------
        list[Publication]
            Parsed publications.
        """
        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }

        context = RequestContext(
            source=self._source_name,
            method="fetch_by_pmids",
            params={"pmid_count": len(pmids)},
        )

        # EFetch returns XML, not JSON - need to handle differently
        await self.rate_limiter.acquire()
        session = await self._get_session()

        logger.info(
            "EFetch request for %d PMIDs",
            len(pmids),
        )

        resp = await session.get(self.EFETCH_URL, params=params)

        if resp.status >= 400:
            body = await resp.text()
            raise DataSourceError(
                self._source_name,
                f"EFetch HTTP {resp.status}: {body[:500]}",
                status_code=resp.status,
            )

        xml_text = await resp.text()
        return self._parse_efetch_xml(xml_text)

    def _build_pubmed_query(
        self,
        drug: str | None,
        condition: str | None,
        date_before: date | None,
    ) -> str:
        """
        Build a PubMed query string from search parameters.

        Uses [tiab] (Title/Abstract) field because:
        - MeSH indexing lags publication by weeks/months
        - New drug names may not have MeSH terms yet
        - Catches recently indexed and epub-ahead-of-print papers

        Parameters
        ----------
        drug : str, optional
            Drug name to search.
        condition : str, optional
            Condition/disease to search.
        date_before : date, optional
            Date cutoff for temporal filtering.

        Returns
        -------
        str
            PubMed query string.

        Raises
        ------
        ValueError
            If neither drug nor condition is provided.
        """
        raise NotImplementedError("_build_pubmed_query not yet implemented")

    def _parse_efetch_xml(self, xml_text: str) -> list[Publication]:
        """
        Parse EFetch XML response into Publication objects.

        Handles both structured abstracts (multiple labeled sections)
        and unstructured abstracts (single text block).

        Parameters
        ----------
        xml_text : str
            Raw XML response from EFetch.

        Returns
        -------
        list[Publication]
            Parsed publications. Malformed articles are skipped.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error("Failed to parse EFetch XML: %s", e)
            raise DataSourceError(self._source_name, f"XML parse error: {e}")

        publications: list[Publication] = []
        for article in root.findall(".//PubmedArticle"):
            pub = self._parse_article(article)
            if pub:
                publications.append(pub)

        return publications

    def _parse_article(self, article_elem: ET.Element) -> Publication | None:
        """
        Parse a single PubmedArticle XML element into a Publication.

        Parameters
        ----------
        article_elem : xml.etree.ElementTree.Element
            A <PubmedArticle> element.

        Returns
        -------
        Publication or None
            Parsed publication, or None if parsing fails.
        """
        try:
            citation = article_elem.find("MedlineCitation")
            if citation is None:
                return None

            # PMID (required)
            pmid_elem = citation.find("PMID")
            if pmid_elem is None or not pmid_elem.text:
                return None
            pmid = pmid_elem.text

            # Article section
            article = citation.find("Article")
            if article is None:
                return None

            # Title (required)
            title_elem = article.find("ArticleTitle")
            if title_elem is None:
                return None
            title = "".join(title_elem.itertext()).strip()
            if not title:
                return None

            # Abstract (optional, empty string if missing)
            abstract = self._parse_abstract(article.find("Abstract"))

            # Journal (required)
            journal_elem = article.find("Journal/ISOAbbreviation")
            if journal_elem is None or not journal_elem.text:
                return None
            journal = journal_elem.text

            # Year (optional, None if missing)
            year = self._parse_year(article)

            # Publication types
            publication_types = self._parse_publication_types(article)

            # MeSH terms
            mesh_terms = self._parse_mesh_terms(citation)

            # DOI (optional)
            doi = self._parse_doi(article_elem)

            return Publication(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                year=year,
                publication_types=publication_types,
                mesh_terms=mesh_terms,
                doi=doi,
            )

        except Exception as e:
            logger.warning("Failed to parse article: %s", e)
            return None

    def _parse_abstract(self, abstract_elem: ET.Element | None) -> str:
        """
        Parse abstract from XML element.

        Handles both structured (multiple labeled sections) and
        unstructured (single text block) abstracts.
        """
        if abstract_elem is None:
            return ""

        abstract_texts = abstract_elem.findall("AbstractText")
        if not abstract_texts:
            return ""

        parts: list[str] = []
        for elem in abstract_texts:
            # Use itertext() to handle nested tags like <i>, <sub>, <sup>
            text = "".join(elem.itertext()).strip()
            label = elem.get("Label")
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)

        return " ".join(parts)

    def _parse_year(self, article_elem: ET.Element) -> int | None:
        """Parse publication year from Article element."""
        # Try JournalIssue/PubDate/Year first
        year_elem = article_elem.find("Journal/JournalIssue/PubDate/Year")
        if year_elem is not None and year_elem.text:
            try:
                return int(year_elem.text)
            except ValueError:
                pass

        # Try MedlineDate as fallback (format: "2024 Jan-Feb" or "2024")
        medline_date = article_elem.find("Journal/JournalIssue/PubDate/MedlineDate")
        if medline_date is not None and medline_date.text:
            # Extract first 4 digits as year
            text = medline_date.text.strip()
            if len(text) >= 4 and text[:4].isdigit():
                return int(text[:4])

        return None

    def _parse_publication_types(self, article_elem: ET.Element) -> list[str]:
        """Parse publication types from Article element."""
        pub_types: list[str] = []
        for pt in article_elem.findall("PublicationTypeList/PublicationType"):
            if pt.text:
                pub_types.append(pt.text)
        return pub_types

    def _parse_mesh_terms(self, citation_elem: ET.Element) -> list[str]:
        """Parse MeSH descriptor names from MedlineCitation element."""
        mesh_terms: list[str] = []
        for heading in citation_elem.findall("MeshHeadingList/MeshHeading"):
            descriptor = heading.find("DescriptorName")
            if descriptor is not None and descriptor.text:
                mesh_terms.append(descriptor.text)
        return mesh_terms

    def _parse_doi(self, article_elem: ET.Element) -> str | None:
        """Parse DOI from PubmedArticle element."""
        pubmed_data = article_elem.find("PubmedData")
        if pubmed_data is None:
            return None

        for article_id in pubmed_data.findall("ArticleIdList/ArticleId"):
            if article_id.get("IdType") == "doi" and article_id.text:
                return article_id.text

        return None
