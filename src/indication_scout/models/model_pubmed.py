"""
Pydantic models for PubMed data.

These are the data contracts between the PubMed client and the agents.
Agents receive these models - they never see raw API responses.
"""

from pydantic import BaseModel


class Publication(BaseModel):
    """A single PubMed article with metadata and abstract."""

    pmid: str  # PubMed identifier (e.g. "38472913")
    title: str
    abstract: str  # joined sections; empty string if missing
    journal: str  # ISO abbreviation (e.g. "N Engl J Med")
    year: int | None  # publication year; None for epub-ahead-of-print
    publication_types: list[str]  # NLM controlled vocabulary
    mesh_terms: list[str]  # MeSH descriptor names only
    doi: str | None = None
