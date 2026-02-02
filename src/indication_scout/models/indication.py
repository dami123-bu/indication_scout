"""Indication data model."""

from dataclasses import dataclass


@dataclass
class Indication:
    """Represents a medical indication/condition."""

    id: str
    name: str
    synonyms: list[str] | None = None
    icd_codes: list[str] | None = None
    mesh_terms: list[str] | None = None
