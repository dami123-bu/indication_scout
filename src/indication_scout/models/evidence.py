from pydantic import BaseModel
from enum import Enum
from datetime import datetime


class EvidenceType(str, Enum):
    MECHANISM = "mechanism"  # drug-target relationship
    GENETIC = "genetic"  # GWAS, rare variant
    LITERATURE = "literature"  # co-mentions, case reports
    CLINICAL_TRIAL = "clinical_trial"  # active or completed trials
    EXPRESSION = "expression"  # tissue/disease expression
    PATHWAY = "pathway"  # shared pathway membership


class EvidenceStrength(str, Enum):
    HIGH = "high"  # RCT, strong genetic, approved indication
    MEDIUM = "medium"  # case series, suggestive genetic
    LOW = "low"  # single case report, computational prediction
    UNKNOWN = "unknown"


class Source(str, Enum):
    OPEN_TARGETS = "open_targets"
    PUBMED = "pubmed"
    CLINICALTRIALS_GOV = "clinicaltrials_gov"
    DRUGBANK = "drugbank"
    CHEMBL = "chembl"


# --- Base Signal ---


class Signal(BaseModel):
    """Base class for all evidence types."""

    id: str
    evidence_type: EvidenceType
    source: Source
    strength: EvidenceStrength
    score: float  # normalized 0-1
    summary: str  # human-readable description
    raw_data: dict | None = None  # original API response
    retrieved_at: datetime


# --- Specific Signal Types ---


class MechanismEvidence(Signal):
    """Drug acts on a target associated with the disease."""

    evidence_type: EvidenceType = EvidenceType.MECHANISM

    drug_id: str
    drug_name: str
    target_id: str
    target_symbol: str
    action_type: str  # inhibitor, agonist, modulator
    disease_id: str
    disease_name: str
    target_disease_score: float  # Open Targets association score
    evidence_sources: list[str]  # genetic, literature, known_drug, etc.


class GeneticEvidence(Signal):
    """Genetic link between target and disease."""

    evidence_type: EvidenceType = EvidenceType.GENETIC

    target_id: str
    target_symbol: str
    disease_id: str
    disease_name: str
    variant_id: str | None
    study_id: str | None
    p_value: float | None
    odds_ratio: float | None
    association_type: str  # gwas, rare_variant, somatic


class LiteratureEvidence(Signal):
    """Published evidence of drug-disease relationship."""

    evidence_type: EvidenceType = EvidenceType.LITERATURE

    pmid: str | None
    pmcid: str | None
    title: str
    journal: str | None
    publication_date: str | None
    publication_type: str  # case_report, case_series, review, rct
    relevant_text: str | None  # key sentence or abstract snippet
    drug_mentioned: str
    disease_mentioned: str


class ClinicalTrialEvidence(Signal):
    """Clinical trial investigating drug for indication."""

    evidence_type: EvidenceType = EvidenceType.CLINICAL_TRIAL

    nct_id: str
    title: str
    status: str  # recruiting, completed, terminated
    phase: str | None  # 1, 2, 3, 4
    enrollment: int | None
    start_date: str | None
    completion_date: str | None
    has_results: bool


class PathwayEvidence(Signal):
    """Drug target and disease share pathway involvement."""

    evidence_type: EvidenceType = EvidenceType.PATHWAY

    pathway_id: str
    pathway_name: str
    drug_target_id: str
    drug_target_symbol: str
    disease_targets: list[str]  # other pathway members linked to disease


# --- Aggregated Signal Container ---


class EvidenceBundle(BaseModel):
    """All evidence for a drug-indication pair."""

    drug_id: str
    drug_name: str
    indication_id: str
    indication_name: str

    mechanism_evidence: list[MechanismEvidence] = []
    genetic_evidence: list[GeneticEvidence] = []
    literature_evidence: list[LiteratureEvidence] = []
    clinical_trial_evidence: list[ClinicalTrialEvidence] = []
    pathway_evidence: list[PathwayEvidence] = []

    @property
    def all_evidence(self) -> list[Signal]:
        return (
            self.mechanism_evidence
            + self.genetic_evidence
            + self.literature_evidence
            + self.clinical_trial_evidence
            + self.pathway_evidence
        )

    @property
    def total_count(self) -> int:
        return len(self.all_evidence)

    @property
    def high_strength_count(self) -> int:
        return sum(1 for e in self.all_evidence if e.strength == EvidenceStrength.HIGH)

    def by_type(self, evidence_type: EvidenceType) -> list[Signal]:
        return [e for e in self.all_evidence if e.evidence_type == evidence_type]


# --- Scoring Output ---


class EvidenceScore(BaseModel):
    """Computed score for a drug-indication pair."""

    overall_score: float  # 0-1 composite
    confidence: str  # high, medium, low

    mechanism_score: float
    genetic_score: float
    literature_score: float
    clinical_score: float

    rationale: str  # LLM-generated explanation
    gaps: list[str]  # what evidence is missing
    recommendations: list[str]  # suggested next steps
