"""Report data model."""

from dataclasses import dataclass, field
from datetime import datetime

from indication_scout.models.drug import Drug
from indication_scout.models.indication import Indication
from indication_scout.models.evidence import Evidence


@dataclass
class Report:
    """Represents a drug repurposing analysis report."""

    id: str
    drug: Drug
    indication: Indication
    evidence: list[Evidence] = field(default_factory=list)
    overall_score: float = 0.0
    recommendation: str = ""
    created_at: datetime = field(default_factory=datetime.now)
