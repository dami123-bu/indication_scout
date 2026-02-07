"""
Pydantic models for Open Targets data.

These are the data contracts between the Open Targets client and the agents.
Agents receive these models — they never see raw GraphQL responses.
"""

from pydantic import BaseModel


class TargetNotFoundError(Exception):
    """Raised when a target_id is not found in DrugEvaluation.targets."""

    def __init__(self, target_id: str):
        self.target_id = target_id
        super().__init__(f"Target '{target_id}' not found")


# ------------------------------------------------------------------
# Target-level models
# ------------------------------------------------------------------


class Association(BaseModel):
    """Target-disease association with per-datatype evidence breakdown."""

    disease_id: str
    disease_name: str
    overall_score: float
    datatype_scores: dict[
        str, float
    ]  # e.g. {"genetic_association": 0.7, "literature": 0.9}
    therapeutic_areas: list[str]


class Pathway(BaseModel):
    """Reactome pathway the target participates in."""

    pathway_id: str
    pathway_name: str
    top_level_pathway: str


class Interaction(BaseModel):
    """Protein-protein interaction partner."""

    interacting_target_id: str
    interacting_target_symbol: str
    interaction_score: float
    source_database: str  # IntAct, STRING, Signor, Reactome
    biological_role: str
    evidence_count: int


class CellTypeExpression(BaseModel):
    """Protein expression in a specific cell type within a tissue."""

    name: str
    level: int
    reliability: bool


class RNAExpression(BaseModel):
    """RNA expression measurement."""

    value: float  # TPM
    quantile: int  # relative level across tissues
    unit: str = "TPM"


class ProteinExpression(BaseModel):
    """Protein expression measurement with cell type detail."""

    level: int  # 0-3
    reliability: bool
    cell_types: list[CellTypeExpression] = []


class TissueExpression(BaseModel):
    """Expression data for a single tissue."""

    tissue_id: str  # UBERON ID
    tissue_name: str
    tissue_anatomical_system: str
    rna: RNAExpression
    protein: ProteinExpression


class BiologicalModel(BaseModel):
    """A specific mouse model (knockout, knock-in, etc.)."""

    allelic_composition: str
    genetic_background: str
    literature: list[str] = []  # PMIDs
    model_id: str  # MGI ID


class MousePhenotype(BaseModel):
    """Phenotype observed in mouse models for this target."""

    phenotype_id: str  # MP ontology ID
    phenotype_label: str
    phenotype_categories: list[str]  # top-level MP categories
    biological_models: list[BiologicalModel] = []


class SafetyEffect(BaseModel):
    """A safety effect with direction and dosing conditions."""

    direction: str
    dosing: str | None = None


class SafetyLiability(BaseModel):
    """Known target safety effect from Open Targets."""

    event: str | None = None
    event_id: str | None = None
    effects: list[SafetyEffect] = []
    datasource: str | None = None
    literature: str | None = None
    url: str | None = None


class AdverseEvent(BaseModel):
    """Significant adverse event from FAERS for a drug."""

    name: str
    meddra_code: str | None = None
    count: int
    log_likelihood_ratio: float


class GeneticConstraint(BaseModel):
    """GnomAD loss-of-function intolerance score."""

    constraint_type: str  # "syn", "mis", "lof"
    oe: float | None = None  # observed/expected
    oe_lower: float | None = None
    oe_upper: float | None = None
    score: float | None = None
    upper_bin: int | None = None  # 0 = most constrained, 5 = least


class KnownDrug(BaseModel):
    """A drug known to act on this target, with indication and phase info."""

    drug_id: str  # ChEMBL ID
    drug_name: str
    disease_id: str
    disease_name: str
    phase: float
    status: str | None = None
    mechanism_of_action: str
    clinical_trial_ids: list[str] = []  # NCT IDs


class TargetData(BaseModel):
    """Everything Open Targets knows about a target. Populated once by load()."""

    target_id: str
    symbol: str
    name: str
    associations: list[Association] = []
    pathways: list[Pathway] = []
    interactions: list[Interaction] = []
    known_drugs: list[KnownDrug] = []
    expressions: list[TissueExpression] = []
    mouse_phenotypes: list[MousePhenotype] = []
    safety_liabilities: list[SafetyLiability] = []
    genetic_constraint: list[GeneticConstraint] = []


# ------------------------------------------------------------------
# Drug-level models
# ------------------------------------------------------------------


class DrugTarget(BaseModel):
    """A target linked to a drug via mechanism of action."""

    target_id: str  # Ensembl gene ID
    target_symbol: str  # e.g. "GLP1R"
    mechanism_of_action: str  # e.g. "Glucagon-like peptide 1 receptor agonist"
    action_type: str | None = None  # e.g. "AGONIST", "INHIBITOR"


class DrugWarning(BaseModel):
    """Black box warning or withdrawal."""

    warning_type: str
    description: str | None = None
    toxicity_class: str | None = None
    country: str | None = None
    year: int | None = None
    efo_id: str | None = None


class Indication(BaseModel):
    """An approved or investigational indication for a drug."""

    disease_id: str
    disease_name: str
    max_phase: float  # 0-4
    references: list[dict] = []


class DrugData(BaseModel):
    """Everything Open Targets knows about a drug. Populated once by load()."""

    chembl_id: str
    name: str
    synonyms: list[str] = []
    trade_names: list[str] = []
    drug_type: str
    is_approved: bool
    max_clinical_phase: float
    year_first_approved: int | None = None
    warnings: list[DrugWarning] = []
    indications: list[Indication] = []
    targets: list[DrugTarget] = []
    adverse_events: list[AdverseEvent] = []
    adverse_events_critical_value: float = 0.0


# ------------------------------------------------------------------
# Safety rollup (convenience wrapper)
# ------------------------------------------------------------------


class TargetSafety(BaseModel):
    """Combined safety data: target safety liabilities + genetic constraint."""

    safety_liabilities: list[SafetyLiability] = []
    genetic_constraint: list[GeneticConstraint] = []


# ------------------------------------------------------------------
# Top-level evaluation blob
# ------------------------------------------------------------------


class DrugEvaluation(BaseModel):
    """
    The complete prefetched data for a drug evaluation.
    One drug, one or more targets, everything cached together.
    Built by OpenTargetsClient.load(), consumed by all agents.
    """

    drug: DrugData
    targets: dict[str, TargetData]  # keyed by Ensembl ID

    def get_target(self, target_id: str) -> TargetData:
        if target_id not in self.targets:
            raise TargetNotFoundError(target_id)
        return self.targets[target_id]

    @property
    def primary_target(self) -> TargetData | None:
        """First target — most drugs have one main target. Returns None if no targets."""
        if not self.drug.targets:
            return None
        first_id = self.drug.targets[0].target_id
        return self.targets.get(first_id)

    @property
    def approved_disease_ids(self) -> set[str]:
        """Disease IDs with phase 4 / approved status — for filtering."""
        return {i.disease_id for i in self.drug.indications if i.max_phase >= 4}

    @property
    def investigated_disease_ids(self) -> set[str]:
        """All disease IDs being actively investigated (any phase)."""
        return {i.disease_id for i in self.drug.indications}
