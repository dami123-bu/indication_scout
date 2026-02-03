"""Drug data model."""

from pydantic import BaseModel

class Drug(BaseModel):
    chembl_id: str
    name: str
    description: str | None = None
    drug_type: str | None = None
    max_clinical_phase: int | None = None
    mechanisms: list[Mechanism] = []
    indications: list[DiseaseIndication] = []
