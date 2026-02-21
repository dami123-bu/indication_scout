"""openFDA FAERS (Drug Adverse Event) data models."""

from pydantic import BaseModel


class FAERSReactionCount(BaseModel):
    """Adverse reaction term and its report count from the FAERS count endpoint."""

    term: str
    count: int


class FAERSEvent(BaseModel):
    """Single adverse event record from FAERS, one row per report."""

    medicinal_product: str
    drug_indication: str | None = None
    reaction: str
    reaction_outcome: str | None = None
    serious: str | None = None
    company_numb: str | None = None