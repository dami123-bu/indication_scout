"""Unit tests for PubMedClient helper functions."""

import os

import pytest

from indication_scout.data_sources.pubmed import PubMedClient

from dotenv import load_dotenv

load_dotenv()

ncbi_api_key = os.getenv("NCBI_API_KEY")

# Sample XML for a complete article
COMPLETE_ARTICLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>38472913</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>N Engl J Med</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Semaglutide for NASH: Phase 3 Results</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Nonalcoholic steatohepatitis is a growing concern.</AbstractText>
          <AbstractText Label="METHODS">We randomly assigned 320 patients.</AbstractText>
          <AbstractText Label="RESULTS">Resolution occurred in 59% of patients.</AbstractText>
          <AbstractText Label="CONCLUSIONS">Semaglutide significantly improved outcomes.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Clinical Trial, Phase III</PublicationType>
          <PublicationType>Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName MajorTopicYN="Y">Non-alcoholic Fatty Liver Disease</DescriptorName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName>Glucagon-Like Peptide-1 Receptor</DescriptorName>
          <QualifierName>agonists</QualifierName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">38472913</ArticleId>
        <ArticleId IdType="doi">10.1056/NEJMoa2312345</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

# Article with unstructured abstract (no labels)
UNSTRUCTURED_ABSTRACT_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>J Test Med</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <Year>2023</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Simple Study Title</ArticleTitle>
        <Abstract>
          <AbstractText>This is a simple unstructured abstract without any labels.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

# Article with no abstract
NO_ABSTRACT_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999999</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>Brief Rep</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Letter to the Editor</ArticleTitle>
        <PublicationTypeList>
          <PublicationType>Letter</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">99999999</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

# Article with MedlineDate instead of Year
MEDLINE_DATE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111111</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>Old J Med</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <MedlineDate>2022 Jan-Feb</MedlineDate>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Study with MedlineDate</ArticleTitle>
        <Abstract>
          <AbstractText>Abstract text here.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">11111111</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

# Article with nested tags in title and abstract
NESTED_TAGS_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222222</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>Chem J</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Effect of H<sub>2</sub>O on <i>in vitro</i> Results</ArticleTitle>
        <Abstract>
          <AbstractText>The compound showed IC<sub>50</sub> of 10<sup>-6</sup> M in <i>Homo sapiens</i> cells.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">22222222</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

# Multiple articles in one response
MULTIPLE_ARTICLES_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111111</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>J One</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>First Article</ArticleTitle>
        <Abstract>
          <AbstractText>First abstract.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">11111111</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222222</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>J Two</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <Year>2023</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Second Article</ArticleTitle>
        <Abstract>
          <AbstractText>Second abstract.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Review</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">22222222</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

# Article missing required fields (no journal)
MISSING_JOURNAL_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>33333333</PMID>
      <Article>
        <Journal>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Article Without Journal Abbreviation</ArticleTitle>
        <Abstract>
          <AbstractText>Abstract here.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestParseEfetchXml:
    """Tests for _parse_efetch_xml method."""

    def setup_method(self):
        self.client = PubMedClient()

    def test_parse_complete_article(self):
        """Should parse a complete article with all fields."""
        publications = self.client._parse_efetch_xml(COMPLETE_ARTICLE_XML)

        assert len(publications) == 1
        pub = publications[0]

        assert pub.pmid == "38472913"
        assert pub.title == "Semaglutide for NASH: Phase 3 Results"
        assert pub.journal == "N Engl J Med"
        assert pub.year == 2024
        assert pub.doi == "10.1056/NEJMoa2312345"
        assert pub.publication_types == [
            "Clinical Trial, Phase III",
            "Randomized Controlled Trial",
        ]
        assert pub.mesh_terms == [
            "Non-alcoholic Fatty Liver Disease",
            "Glucagon-Like Peptide-1 Receptor",
        ]

        # Structured abstract should be joined with labels
        assert (
            "BACKGROUND: Nonalcoholic steatohepatitis is a growing concern."
            in pub.abstract
        )
        assert "METHODS: We randomly assigned 320 patients." in pub.abstract
        assert "RESULTS: Resolution occurred in 59% of patients." in pub.abstract
        assert (
            "CONCLUSIONS: Semaglutide significantly improved outcomes." in pub.abstract
        )

    def test_parse_unstructured_abstract(self):
        """Should parse article with unstructured abstract (no labels)."""
        publications = self.client._parse_efetch_xml(UNSTRUCTURED_ABSTRACT_XML)

        assert len(publications) == 1
        pub = publications[0]

        assert pub.pmid == "12345678"
        assert (
            pub.abstract == "This is a simple unstructured abstract without any labels."
        )

    def test_parse_no_abstract(self):
        """Should return empty string for articles without abstract."""
        publications = self.client._parse_efetch_xml(NO_ABSTRACT_XML)

        assert len(publications) == 1
        pub = publications[0]

        assert pub.pmid == "99999999"
        assert pub.abstract == ""
        assert pub.publication_types == ["Letter"]

    def test_parse_medline_date(self):
        """Should extract year from MedlineDate when Year element is missing."""
        publications = self.client._parse_efetch_xml(MEDLINE_DATE_XML)

        assert len(publications) == 1
        pub = publications[0]

        assert pub.pmid == "11111111"
        assert pub.year == 2022

    def test_parse_nested_tags(self):
        """Should handle nested tags in title and abstract using itertext()."""
        publications = self.client._parse_efetch_xml(NESTED_TAGS_XML)

        assert len(publications) == 1
        pub = publications[0]

        assert pub.pmid == "22222222"
        # itertext() should extract all text including nested elements
        assert "H2O" in pub.title
        assert "in vitro" in pub.title
        assert "IC50" in pub.abstract
        assert "10-6" in pub.abstract
        assert "Homo sapiens" in pub.abstract

    def test_parse_multiple_articles(self):
        """Should parse multiple articles from single response."""
        publications = self.client._parse_efetch_xml(MULTIPLE_ARTICLES_XML)

        assert len(publications) == 2

        pmids = [p.pmid for p in publications]
        assert "11111111" in pmids
        assert "22222222" in pmids

        # Find each article and verify
        first = next(p for p in publications if p.pmid == "11111111")
        assert first.title == "First Article"
        assert first.journal == "J One"
        assert first.year == 2024

        second = next(p for p in publications if p.pmid == "22222222")
        assert second.title == "Second Article"
        assert second.journal == "J Two"
        assert second.year == 2023
        assert second.publication_types == ["Review"]

    def test_parse_skips_malformed_articles(self):
        """Should skip articles missing required fields."""
        publications = self.client._parse_efetch_xml(MISSING_JOURNAL_XML)

        # Article without ISOAbbreviation should be skipped
        assert len(publications) == 0

    def test_parse_empty_response(self):
        """Should return empty list for empty PubmedArticleSet."""
        xml = """<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>"""
        publications = self.client._parse_efetch_xml(xml)

        assert publications == []


class TestParseArticleEdgeCases:
    """Tests for edge cases in _parse_article."""

    def setup_method(self):
        self.client = PubMedClient()

    def test_article_without_mesh_terms(self):
        """Should return empty mesh_terms list when MeshHeadingList is missing."""
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>44444444</PMID>
              <Article>
                <Journal>
                  <ISOAbbreviation>J Test</ISOAbbreviation>
                  <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
                </Journal>
                <ArticleTitle>No MeSH Terms</ArticleTitle>
                <Abstract><AbstractText>Abstract.</AbstractText></Abstract>
                <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
              </Article>
            </MedlineCitation>
            <PubmedData>
              <ArticleIdList><ArticleId IdType="pubmed">44444444</ArticleId></ArticleIdList>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        publications = self.client._parse_efetch_xml(xml)

        assert len(publications) == 1
        assert publications[0].mesh_terms == []

    def test_article_without_doi(self):
        """Should return None for doi when not present."""
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>55555555</PMID>
              <Article>
                <Journal>
                  <ISOAbbreviation>J Test</ISOAbbreviation>
                  <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
                </Journal>
                <ArticleTitle>No DOI</ArticleTitle>
                <Abstract><AbstractText>Abstract.</AbstractText></Abstract>
                <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
              </Article>
            </MedlineCitation>
            <PubmedData>
              <ArticleIdList><ArticleId IdType="pubmed">55555555</ArticleId></ArticleIdList>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        publications = self.client._parse_efetch_xml(xml)

        assert len(publications) == 1
        assert publications[0].doi is None

    def test_article_without_publication_types(self):
        """Should return empty publication_types list when missing."""
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>66666666</PMID>
              <Article>
                <Journal>
                  <ISOAbbreviation>J Test</ISOAbbreviation>
                  <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
                </Journal>
                <ArticleTitle>No Pub Types</ArticleTitle>
                <Abstract><AbstractText>Abstract.</AbstractText></Abstract>
              </Article>
            </MedlineCitation>
            <PubmedData>
              <ArticleIdList><ArticleId IdType="pubmed">66666666</ArticleId></ArticleIdList>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        publications = self.client._parse_efetch_xml(xml)

        assert len(publications) == 1
        assert publications[0].publication_types == []

    def test_article_without_year(self):
        """Should return None for year when both Year and MedlineDate are missing."""
        xml = """<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>77777777</PMID>
              <Article>
                <Journal>
                  <ISOAbbreviation>J Test</ISOAbbreviation>
                  <JournalIssue><PubDate></PubDate></JournalIssue>
                </Journal>
                <ArticleTitle>No Year</ArticleTitle>
                <Abstract><AbstractText>Abstract.</AbstractText></Abstract>
                <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
              </Article>
            </MedlineCitation>
            <PubmedData>
              <ArticleIdList><ArticleId IdType="pubmed">77777777</ArticleId></ArticleIdList>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>
        """
        publications = self.client._parse_efetch_xml(xml)

        assert len(publications) == 1
        assert publications[0].year is None


class TestFetchByPmidsEdgeCases:
    """Tests for fetch_by_pmids edge case handling."""

    def setup_method(self):
        self.client = PubMedClient()

    @pytest.mark.asyncio
    async def test_empty_pmids_returns_empty_list(self):
        """Should return empty list for empty input without API call."""
        result = await self.client.fetch_by_pmids([])
        assert result == []

    @pytest.mark.asyncio
    async def test_deduplicates_pmids(self):
        """Should deduplicate PMIDs before processing."""
        # We can't easily test this without mocking, but we can verify
        # the deduplication logic by checking it doesn't raise
        # This test would need integration or mocking to fully verify
        pass


class TestBuildPubmedQuery:
    """Tests for _build_pubmed_query method."""

    def setup_method(self):
        self.client = PubMedClient()

    def test_drug_only(self):
        """Should build query with drug in title/abstract."""
        query = self.client._build_pubmed_query(
            drug="semaglutide",
            condition=None,
            date_before=None,
        )
        assert query == "semaglutide[tiab]"

    def test_condition_only(self):
        """Should build query with condition in title/abstract."""
        query = self.client._build_pubmed_query(
            drug=None,
            condition="NASH",
            date_before=None,
        )
        assert query == "NASH[tiab]"

    def test_drug_and_condition(self):
        """Should build query with both drug and condition."""
        query = self.client._build_pubmed_query(
            drug="semaglutide",
            condition="NASH",
            date_before=None,
        )
        assert query == "semaglutide[tiab] AND NASH[tiab]"

    def test_with_date_before(self):
        """Should include date range filter when date_before is provided."""
        from datetime import date

        query = self.client._build_pubmed_query(
            drug="semaglutide",
            condition="obesity",
            date_before=date(2023, 6, 15),
        )
        assert (
            query
            == 'semaglutide[tiab] AND obesity[tiab] AND "1900/01/01":"2023/06/15"[pdat]'
        )

    def test_drug_only_with_date(self):
        """Should work with drug only and date filter."""
        from datetime import date

        query = self.client._build_pubmed_query(
            drug="tirzepatide",
            condition=None,
            date_before=date(2022, 1, 1),
        )
        assert query == 'tirzepatide[tiab] AND "1900/01/01":"2022/01/01"[pdat]'

    def test_neither_drug_nor_condition_raises(self):
        """Should raise ValueError if neither drug nor condition provided."""
        with pytest.raises(
            ValueError, match="At least one of drug or condition is required"
        ):
            self.client._build_pubmed_query(
                drug=None,
                condition=None,
                date_before=None,
            )


class TestClientConfiguration:
    """Tests for PubMedClient configuration."""

    def test_rate_limit_without_api_key(self):
        """Should set rate limit to 3/s without API key."""
        client = PubMedClient()
        assert client.rate_limiter.rate == 3.0

    def test_rate_limit_with_api_key(self):
        """Should set rate limit to 10/s with API key."""
        client = PubMedClient(api_key=ncbi_api_key)
        assert client.rate_limiter.rate == 10.0

    def test_base_params_without_api_key(self):
        """Should include tool and email in base params."""
        client = PubMedClient(email="test@example.com", tool="testapp")
        params = client._base_params()

        assert params["tool"] == "testapp"
        assert params["email"] == "test@example.com"
        assert "ncbi_api_key" not in params

    def test_base_params_with_api_key(self):
        """Should include api_key in base params when provided."""
        client = PubMedClient(
            api_key=ncbi_api_key, email="test@example.com", tool="testapp"
        )
        params = client._base_params()

        assert params["tool"] == "testapp"
        assert params["email"] == "test@example.com"
        assert params["api_key"] == ncbi_api_key

    def test_source_name(self):
        """Should return 'pubmed' as source name."""
        client = PubMedClient()
        assert client._source_name == "pubmed"
