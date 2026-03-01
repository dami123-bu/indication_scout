"""Unit tests for PubMedClient."""

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.pubmed import PubMedClient

# --- _parse_pubmed_xml ---


def test_invalid_xml_raises_error():
    """Test that invalid XML raises DataSourceError."""
    client = PubMedClient()
    invalid_xml = "not valid xml <unclosed"

    with pytest.raises(DataSourceError) as exc_info:
        client._parse_pubmed_xml(invalid_xml)

    assert exc_info.value.source == "pubmed"
    assert "Failed to parse XML" in str(exc_info.value)


def test_empty_xml_raises_error():
    """Test that empty string raises DataSourceError."""
    client = PubMedClient()

    with pytest.raises(DataSourceError) as exc_info:
        client._parse_pubmed_xml("")

    assert exc_info.value.source == "pubmed"
    assert "Failed to parse XML" in str(exc_info.value)


def test_valid_xml_no_articles_returns_empty_list():
    """Test that valid XML with no PubmedArticle elements returns empty list."""
    client = PubMedClient()
    xml = "<PubmedArticleSet></PubmedArticleSet>"

    result = client._parse_pubmed_xml(xml)

    assert result == []


def test_valid_xml_parses_article():
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


def test_article_without_pmid_is_skipped():
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
