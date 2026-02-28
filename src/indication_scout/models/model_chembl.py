"""Pydantic models for ChEMBL API data."""

from pydantic import BaseModel


class MoleculeData(BaseModel):
    """Data returned by the ChEMBL molecule endpoint for a single compound."""

    molecule_chembl_id: str
    molecule_type: str
    max_phase: str | None
    atc_classifications: list[str]
    black_box_warning: int
    first_approval: int | None
    oral: bool
