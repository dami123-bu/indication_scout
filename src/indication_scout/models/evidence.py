"""Evidence data model and scoring."""

from dataclasses import dataclass, field
from enum import Enum


class EvidenceType(Enum):
    """Types of evidence for drug-indication relationships."""

    CLINICAL_TRIAL = "clinical_trial"
    LITERATURE = "literature"
    MECHANISM = "mechanism"
    REAL_WORLD = "real_world"
    SAFETY = "safety"


class EvidenceStrength(Enum):
    """Strength levels for evidence."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    ANECDOTAL = "anecdotal"


@dataclass
class Evidence:
    """Represents evidence for a drug-indication relationship."""

    id: str
    type: EvidenceType
    strength: EvidenceStrength
    source: str
    summary: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)
