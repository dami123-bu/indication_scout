"""Unit tests for the SQLAlchemy DB model and session layer."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import sqlalchemy as sa

from indication_scout.sqlalchemy.pubmed_abstracts import PubmedAbstracts

# --- PubmedAbstracts model ---


def test_pubmed_abstracts_tablename():
    """PubmedAbstracts should map to the pubmed_abstracts table."""
    assert PubmedAbstracts.__tablename__ == "pubmed_abstracts"


def test_pubmed_abstracts_primary_key():
    """pmid should be the primary key."""
    pk_cols = [col.name for col in PubmedAbstracts.__table__.primary_key]
    assert pk_cols == ["pmid"]


def test_pubmed_abstracts_columns():
    """PubmedAbstracts table should have exactly the expected columns."""
    col_names = {col.name for col in PubmedAbstracts.__table__.columns}
    assert col_names == {
        "pmid",
        "title",
        "abstract",
        "authors",
        "journal",
        "pub_date",
        "mesh_terms",
        "embedding",
        "fetched_at",
    }


def test_pubmed_abstracts_nullable_constraints():
    """Only pmid and title should be non-nullable."""
    cols = {col.name: col for col in PubmedAbstracts.__table__.columns}
    assert cols["pmid"].nullable is False
    assert cols["title"].nullable is False
    assert cols["abstract"].nullable is True
    assert cols["authors"].nullable is True
    assert cols["journal"].nullable is True
    assert cols["pub_date"].nullable is True
    assert cols["mesh_terms"].nullable is True
    assert cols["embedding"].nullable is True
    assert cols["fetched_at"].nullable is False


def test_pubmed_abstracts_all_fields():
    """PubmedAbstracts should store all fields correctly."""
    now = datetime(2024, 6, 1, 12, 0, 0)
    record = PubmedAbstracts(
        pmid="12345678",
        title="Metformin and colorectal cancer: a meta-analysis",
        abstract="This meta-analysis evaluates metformin use in colorectal cancer.",
        authors=["Smith, J", "Jones, A"],
        journal="Journal of Clinical Oncology",
        pub_date="2023-04",
        mesh_terms=["Metformin", "Colorectal Neoplasms", "Antineoplastic Agents"],
        embedding=[0.1] * 768,
        fetched_at=now,
    )
    assert record.pmid == "12345678"
    assert record.title == "Metformin and colorectal cancer: a meta-analysis"
    assert (
        record.abstract
        == "This meta-analysis evaluates metformin use in colorectal cancer."
    )
    assert record.authors == ["Smith, J", "Jones, A"]
    assert record.journal == "Journal of Clinical Oncology"
    assert record.pub_date == "2023-04"
    assert record.mesh_terms == [
        "Metformin",
        "Colorectal Neoplasms",
        "Antineoplastic Agents",
    ]
    assert len(record.embedding) == 768
    assert record.embedding[0] == 0.1
    assert record.fetched_at == now


def test_pubmed_abstracts_minimal_fields():
    """PubmedAbstracts should work with only pmid and title."""
    record = PubmedAbstracts(
        pmid="99999999",
        title="Minimal record",
    )
    assert record.pmid == "99999999"
    assert record.title == "Minimal record"
    assert record.abstract is None
    assert record.authors is None
    assert record.journal is None
    assert record.pub_date is None
    assert record.mesh_terms is None
    assert record.embedding is None


def test_pubmed_abstracts_embedding_dimension():
    """Embedding column should be configured for 768 dimensions."""
    embedding_col = PubmedAbstracts.__table__.columns["embedding"]
    assert embedding_col.type.dim == 768


# --- session ---


def test_get_db_yields_and_closes():
    """get_db should yield a session and close it on exit."""
    mock_session = MagicMock()
    mock_factory = MagicMock(return_value=mock_session)

    with patch(
        "indication_scout.db.session._make_session_factory",
        return_value=mock_factory,
    ):
        from indication_scout.db.session import get_db

        gen = get_db()
        session = next(gen)
        assert session is mock_session

        try:
            next(gen)
        except StopIteration:
            pass

        mock_session.close.assert_called_once()
