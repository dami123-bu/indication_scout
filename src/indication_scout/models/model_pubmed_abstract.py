"""PubMed data models."""

from pydantic import BaseModel


class PubmedAbstract(BaseModel):
    """Parsed PubMed abstract data."""

    pmid: str
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = []
    journal: str | None = None
    pub_date: str | None = None
    mesh_terms: list[str] = []
    keywords: list[str] = []
