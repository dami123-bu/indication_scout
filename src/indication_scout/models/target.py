"""Target data model."""

from pydantic import BaseModel


class Target(BaseModel):
    """A gene or protein
    Potentially, a drug acts on it.

    Identified by a source-independent ID, an optional NCBI gene ID, an optional
    Ensembl gene ID (e.g. ENSG00000146648), and a human-readable symbol (e.g. EGFR).
    Referenced by DrugActivity to represent the molecular target of a drug's mechanism.
    """

    id: str
    ncbi_id: str | None = None
    ensembl_id: str | None = None  # Ensembl gene ID, e.g. "ENSG00000146648"
    symbol: str  # gene symbol, e.g. "EGFR"
