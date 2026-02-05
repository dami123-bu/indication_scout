"""Indication data models."""

from pydantic import BaseModel, Field
from uuid import uuid4


class Indication(BaseModel):
    """Anything a drug can be used to treat.

    This is the base class for all treatable conditions — diseases, symptoms,
    syndromes, or other clinical presentations (e.g. "fever", "puffy eyes").
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str


class DiseaseIndication(Indication):
    """A disease entity with external ontology cross-references.

    Extends Indication with identifiers from disease ontologies
    (EFO from Open Targets, MONDO from Monarch Initiative).
    """

    efo_id: str | None = None
    mondo_id: str | None = None
    # Broad disease categories
    # e.g. 'coronary artery disease' belongs to therapeutic area 'cardiovascular disease'
    therapeutic_areas: list[str] = []
