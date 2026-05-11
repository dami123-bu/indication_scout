"""Unit tests for EuropePMCClient response parsing and query construction."""

from datetime import date

import pytest

from indication_scout.data_sources.europe_pmc import EuropePMCClient


# --- _parse_result ---


def test_parse_result_full_record():
    """Map a full Europe PMC `core` record onto every PubmedAbstract field."""
    raw = {
        "pmid": "12345678",
        "title": "Repurposing metformin for cancer prevention.",
        "abstractText": "Background: metformin. Methods: cohort.",
        "authorList": {
            "author": [
                {"fullName": "Smith J", "lastName": "Smith"},
                {"fullName": "Doe A", "lastName": "Doe"},
            ]
        },
        "journalInfo": {"journal": {"title": "J Onc"}},
        "firstPublicationDate": "2024-05-01",
        "meshHeadingList": {
            "meshHeading": [
                {"descriptorName": "Metformin"},
                {"descriptorName": "Neoplasms"},
            ]
        },
        "keywordList": {"keyword": ["repurposing", "cancer"]},
    }

    article = EuropePMCClient._parse_result(raw)

    assert article is not None
    assert article.pmid == "12345678"
    assert article.title == "Repurposing metformin for cancer prevention."
    assert article.abstract == "Background: metformin. Methods: cohort."
    assert article.authors == ["Smith J", "Doe A"]
    assert article.journal == "J Onc"
    assert article.pub_date == "2024-05-01"
    assert article.mesh_terms == ["Metformin", "Neoplasms"]
    assert article.keywords == ["repurposing", "cancer"]


def test_parse_result_missing_pmid_returns_none():
    """Records without a PMID are skipped so PubmedAbstracts PK stays valid."""
    assert EuropePMCClient._parse_result({"id": "PPR12345", "title": "x"}) is None


def test_parse_result_minimal_record():
    """A record missing optional nested objects parses with defaults applied."""
    article = EuropePMCClient._parse_result({"pmid": "1", "title": "T"})

    assert article is not None
    assert article.pmid == "1"
    assert article.title == "T"
    assert article.abstract is None
    assert article.authors == []
    assert article.journal is None
    assert article.pub_date is None
    assert article.mesh_terms == []
    assert article.keywords == []


# --- _augment_query ---


@pytest.mark.parametrize(
    "query, date_before, expected",
    [
        ("metformin AND obesity", None, "(metformin AND obesity) AND SRC:MED"),
        (
            "metformin",
            date(2025, 1, 1),
            "(metformin) AND SRC:MED AND FIRST_PDATE:[1900-01-01 TO 2024-12-31]",
        ),
    ],
)
def test_augment_query(query, date_before, expected):
    assert EuropePMCClient._augment_query(query, date_before) == expected
