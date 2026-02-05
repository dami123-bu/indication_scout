"""Drug data models."""

from __future__ import annotations

from pydantic import BaseModel, Field
from uuid import uuid4

from indication_scout.models import Indication
from indication_scout.models.target import Target


class Drug(BaseModel):
    """A pharmaceutical compound.

    Identified by a source-independent ID and optionally by external database
    IDs (ChEMBL, DrugBank). Each Drug has zero or more DrugActivity entries
    linking it to the indications it acts on via specific mechanisms.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    chembl_id: str | None = None
    drugbank_id: str | None = None
    generic_name: str
    brand_name: str | None = None
    description: str | None = None
    drug_type: str | None = None
    activities: list[DrugActivity] = []
    is_approved: bool = False
    has_been_withdrawn: bool = False
    year_first_approved: int | None = None
    max_clinical_phase: float | None = None
    synonyms: list[str] = []
    trade_names: list[str] = []
    parent_molecule_id: str | None = None
    child_molecule_ids: list[str] = []


class DrugActivity(BaseModel):
    """A drug's mechanism of action on a target for a specific indication.

    All fields are optional to allow partially populated entries from
    different data sources.  A single Drug can have multiple DrugActivity
    entries — one per mechanism-target-indication combination.
    """

    description: str | None = None
    action_type: str | None = None
    target: Target | None = None
    indication: Indication | None = None
