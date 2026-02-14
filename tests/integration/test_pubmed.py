"""Integration tests for PubMedClient."""

import unittest

from indication_scout.data_sources.pubmed import PubMedClient


class TestPubMedClient(unittest.IsolatedAsyncioTestCase):
    """Integration tests for PubMedClient."""

    async def asyncSetUp(self):
        self.client = PubMedClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_search_returns_pmids(self):
        """Test search returns a list of PMIDs for a query."""
        pmids = await self.client.search("semaglutide diabetes", max_results=10)

        self.assertEqual(len(pmids), 10)
        # PMIDs are numeric strings
        for pmid in pmids:
            self.assertTrue(pmid.isdigit())

    async def test_get_count_returns_total(self):
        """Test get_count returns total count without fetching articles."""
        count = await self.client.get_count("semaglutide diabetes")

        # Should be many articles about semaglutide and diabetes
        self.assertGreater(count, 500)

    async def test_fetch_articles_parses_correctly(self):
        """Test fetch_articles returns parsed PubMedArticle objects."""
        # First search for a known article
        pmids = await self.client.search(
            "semaglutide obesity clinical trial", max_results=5
        )
        articles = await self.client.fetch_articles(pmids)

        self.assertEqual(len(articles), 5)

        # Verify article structure
        for article in articles:
            self.assertTrue(article.pmid.isdigit())
            self.assertIsNotNone(article.title)
            self.assertIsInstance(article.authors, list)

    async def test_fetch_specific_article(self):
        """Test fetching a specific known article by PMID."""
        # PMID 27633186 - SUSTAIN-6 semaglutide cardiovascular trial (NEJM 2016)
        articles = await self.client.fetch_articles(["27633186"])

        self.assertEqual(len(articles), 1)
        article = articles[0]

        self.assertEqual(article.pmid, "27633186")
        self.assertIn("semaglutide", article.title.lower())
        self.assertEqual(article.journal, "The New England journal of medicine")
        self.assertGreater(len(article.authors), 0)
        # First author is Marso
        self.assertTrue(article.authors[0].startswith("Marso"))

    async def test_fetch_articles_empty_list_returns_empty(self):
        """Test that fetching empty list returns empty list."""
        articles = await self.client.fetch_articles([])

        self.assertEqual(articles, [])


class TestPubMedClientEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Tests for edge cases and weird inputs."""

    async def asyncSetUp(self):
        self.client = PubMedClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_search_nonexistent_term_returns_empty(self):
        """Test that a nonexistent search term returns empty list."""
        pmids = await self.client.search(
            "xyzzy_fake_drug_99999_not_real_term", max_results=10
        )

        self.assertEqual(pmids, [])

    async def test_get_count_nonexistent_term_returns_zero(self):
        """Test that a nonexistent term returns zero count."""
        count = await self.client.get_count("xyzzy_fake_drug_99999_not_real_term")

        self.assertEqual(count, 0)

    async def test_search_empty_query_returns_empty(self):
        """Test that empty query returns empty list."""
        pmids = await self.client.search("", max_results=10)

        # Empty search returns nothing
        self.assertEqual(pmids, [])

    async def test_fetch_articles_invalid_pmid_returns_empty(self):
        """Test that invalid PMIDs return empty list (no matching articles)."""
        articles = await self.client.fetch_articles(["99999999999"])

        # Invalid PMID returns no articles
        self.assertEqual(articles, [])

    async def test_search_special_characters_returns_empty(self):
        """Test that special characters in query returns empty list."""
        pmids = await self.client.search("!!!@@@###$$$", max_results=10)

        self.assertEqual(pmids, [])
