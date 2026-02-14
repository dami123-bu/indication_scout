"""Unit tests for PubMedClient."""

from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.pubmed import PubMedClient


class TestParsePubmedXml:
    """Tests for _parse_pubmed_xml method."""

    def test_invalid_xml_raises_error(self):
        """Test that invalid XML raises DataSourceError."""
        client = PubMedClient()
        invalid_xml = "not valid xml <unclosed"

        with pytest.raises(DataSourceError) as exc_info:
            client._parse_pubmed_xml(invalid_xml)

        assert exc_info.value.source == "pubmed"
        assert "Failed to parse XML" in str(exc_info.value)

    def test_empty_xml_raises_error(self):
        """Test that empty string raises DataSourceError."""
        client = PubMedClient()

        with pytest.raises(DataSourceError) as exc_info:
            client._parse_pubmed_xml("")

        assert exc_info.value.source == "pubmed"
        assert "Failed to parse XML" in str(exc_info.value)

    def test_valid_xml_no_articles_returns_empty_list(self):
        """Test that valid XML with no PubmedArticle elements returns empty list."""
        client = PubMedClient()
        xml = "<PubmedArticleSet></PubmedArticleSet>"

        result = client._parse_pubmed_xml(xml)

        assert result == []

    def test_valid_xml_parses_article(self):
        """Test that valid XML with article is parsed correctly."""
        client = PubMedClient()
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Article Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>This is the abstract text.</AbstractText>
                        </Abstract>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                        </AuthorList>
                        <Journal>
                            <Title>Test Journal</Title>
                        </Journal>
                    </Article>
                </MedlineCitation>
                <PubmedData>
                    <History>
                        <PubMedPubDate PubStatus="pubmed">
                            <Year>2023</Year>
                            <Month>06</Month>
                            <Day>15</Day>
                        </PubMedPubDate>
                    </History>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """

        result = client._parse_pubmed_xml(xml)

        assert len(result) == 1
        article = result[0]
        assert article.pmid == "12345678"
        assert article.title == "Test Article Title"
        assert article.abstract == "This is the abstract text."
        assert article.authors == ["Smith, John"]
        assert article.journal == "Test Journal"

    def test_article_without_pmid_is_skipped(self):
        """Test that articles without PMID are skipped."""
        client = PubMedClient()
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>No PMID Article</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """

        result = client._parse_pubmed_xml(xml)

        assert result == []


FETCH_ARTICLES_XML = """\
<?xml version="1.0"?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>11111111</PMID>
            <Article>
                <ArticleTitle>Semaglutide in NASH: Phase 2 results</ArticleTitle>
                <Abstract>
                    <AbstractText Label="BACKGROUND">NASH is prevalent.</AbstractText>
                    <AbstractText Label="RESULTS">Semaglutide improved fibrosis.</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Newsome</LastName>
                        <ForeName>Philip</ForeName>
                    </Author>
                    <Author>
                        <LastName>Harrison</LastName>
                        <ForeName>Stephen</ForeName>
                    </Author>
                </AuthorList>
                <Journal>
                    <Title>The New England Journal of Medicine</Title>
                    <JournalIssue>
                        <PubDate>
                            <Year>2021</Year>
                            <Month>03</Month>
                            <Day>25</Day>
                        </PubDate>
                    </JournalIssue>
                </Journal>
            </Article>
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>Fatty Liver</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Liver Cirrhosis</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
            <KeywordList>
                <Keyword>NASH</Keyword>
                <Keyword>semaglutide</Keyword>
                <Keyword>fibrosis</Keyword>
            </KeywordList>
        </MedlineCitation>
    </PubmedArticle>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>22222222</PMID>
            <Article>
                <ArticleTitle>GLP-1 receptor agonists review</ArticleTitle>
                <Abstract>
                    <AbstractText>Comprehensive review of GLP-1 RAs.</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Drucker</LastName>
                    </Author>
                </AuthorList>
                <Journal>
                    <Title>Nature Reviews Drug Discovery</Title>
                    <JournalIssue>
                        <PubDate>
                            <Year>2024</Year>
                            <Month>01</Month>
                        </PubDate>
                    </JournalIssue>
                </Journal>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>
"""


class TestFetchArticles:
    """Tests for PubMedClient.fetch_articles."""

    @pytest.mark.asyncio
    async def test_fetch_articles(self):
        """fetch_articles should call _run_xml_query and return parsed PubMedArticle list."""
        client = PubMedClient()
        client._run_xml_query = AsyncMock(return_value=FETCH_ARTICLES_XML)

        result = await client.fetch_articles(["11111111", "22222222"])

        # Verify _run_xml_query was called with correct params
        client._run_xml_query.assert_called_once_with(
            client.FETCH_URL,
            {
                "db": "pubmed",
                "id": "11111111,22222222",
                "retmode": "xml",
                "rettype": "abstract",
            },
        )

        assert len(result) == 2

        # First article: multi-section abstract, two authors, MeSH terms, keywords
        art1 = result[0]
        assert art1.pmid == "11111111"
        assert art1.title == "Semaglutide in NASH: Phase 2 results"
        assert (
            art1.abstract
            == "BACKGROUND: NASH is prevalent. RESULTS: Semaglutide improved fibrosis."
        )
        assert art1.authors == ["Newsome, Philip", "Harrison, Stephen"]
        assert art1.journal == "The New England Journal of Medicine"
        assert art1.pub_date == "2021-03-25"
        assert art1.mesh_terms == ["Fatty Liver", "Liver Cirrhosis"]
        assert art1.keywords == ["NASH", "semaglutide", "fibrosis"]

        # Second article: single abstract, author with no ForeName, no MeSH/keywords
        art2 = result[1]
        assert art2.pmid == "22222222"
        assert art2.title == "GLP-1 receptor agonists review"
        assert art2.abstract == "Comprehensive review of GLP-1 RAs."
        assert art2.authors == ["Drucker"]
        assert art2.journal == "Nature Reviews Drug Discovery"
        assert art2.pub_date == "2024-01"
        assert art2.mesh_terms == []
        assert art2.keywords == []

    @pytest.mark.asyncio
    async def test_fetch_articles_empty_pmids(self):
        """fetch_articles should return empty list for empty pmids without calling API."""
        client = PubMedClient()
        client._run_xml_query = AsyncMock()

        result = await client.fetch_articles([])

        assert result == []
        client._run_xml_query.assert_not_called()
