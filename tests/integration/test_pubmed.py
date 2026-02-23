"""Integration tests for PubMedClient."""

import pytest

# --- Main functionality ---


@pytest.mark.asyncio
async def test_search_returns_pmids(pubmed_client):
    """Test search returns a list of PMIDs for a query."""
    pmids = await pubmed_client.search("semaglutide diabetes", max_results=10)

    assert len(pmids) == 10
    # PMIDs are numeric strings
    for pmid in pmids:
        assert pmid.isdigit()


@pytest.mark.asyncio
async def test_get_count_returns_total(pubmed_client):
    """Test get_count returns total count without fetching articles."""
    count = await pubmed_client.get_count("semaglutide diabetes")

    # Should be many articles about semaglutide and diabetes
    assert count > 500


@pytest.mark.asyncio
async def test_fetch_articles_parses_correctly(pubmed_client):
    """Test fetch_articles returns parsed PubMedArticle objects."""
    # First search for a known article
    pmids = await pubmed_client.search(
        "semaglutide obesity clinical trial", max_results=5
    )
    articles = await pubmed_client.fetch_articles(pmids)

    assert len(articles) == 5

    # Verify article structure
    for article in articles:
        assert article.pmid.isdigit()
        assert article.title is not None
        assert isinstance(article.authors, list)


@pytest.mark.asyncio
async def test_fetch_specific_article(pubmed_client):
    """Test fetching a specific known article by PMID."""
    # PMID 27633186 - SUSTAIN-6 semaglutide cardiovascular trial (NEJM 2016)
    articles = await pubmed_client.fetch_articles(["27633186"])

    assert len(articles) == 1
    article = articles[0]

    assert article.pmid == "27633186"
    assert "semaglutide" in article.title.lower()
    assert article.journal == "The New England journal of medicine"
    assert len(article.authors) > 0
    # First author is Marso
    assert article.authors[0].startswith("Marso")


@pytest.mark.asyncio
async def test_fetch_articles_empty_list_returns_empty(pubmed_client):
    """Test that fetching empty list returns empty list."""
    articles = await pubmed_client.fetch_articles([])

    assert articles == []


# --- Edge cases and weird inputs ---


@pytest.mark.asyncio
async def test_search_nonexistent_term_returns_empty(pubmed_client):
    """Test that a nonexistent search term returns empty list."""
    pmids = await pubmed_client.search(
        "xyzzy_fake_drug_99999_not_real_term", max_results=10
    )

    assert pmids == []


@pytest.mark.asyncio
async def test_get_count_nonexistent_term_returns_zero(pubmed_client):
    """Test that a nonexistent term returns zero count."""
    count = await pubmed_client.get_count("xyzzy_fake_drug_99999_not_real_term")

    assert count == 0


@pytest.mark.asyncio
async def test_search_empty_query_returns_empty(pubmed_client):
    """Test that empty query returns empty list."""
    pmids = await pubmed_client.search("", max_results=10)

    # Empty search returns nothing
    assert pmids == []


@pytest.mark.asyncio
async def test_fetch_articles_invalid_pmid_returns_empty(pubmed_client):
    """Test that invalid PMIDs return empty list (no matching articles)."""
    articles = await pubmed_client.fetch_articles(["99999999999"])

    # Invalid PMID returns no articles
    assert articles == []


@pytest.mark.asyncio
async def test_search_special_characters_returns_empty(pubmed_client):
    """Test that special characters in query returns empty list."""
    pmids = await pubmed_client.search("!!!@@@###$$$", max_results=10)

    assert pmids == []
