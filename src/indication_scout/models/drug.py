"""Drug data model."""

from dataclasses import dataclass


@dataclass
class Drug:
    """Represents a drug/compound."""

    id: str
    name: str
    synonyms: list[str] | None = None
    drugbank_id: str | None = None
    chembl_id: str | None = None
    mechanism_of_action: str | None = None
