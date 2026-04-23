"""
Pydantic models for ClinicalTrials.gov data.

These are the data contracts between the ClinicalTrials.gov client and the agents.
Agents never see raw API responses.
"""

from pydantic import BaseModel, model_validator

# ------------------------------------------------------------------
# Trial-level models
# ------------------------------------------------------------------


class Intervention(BaseModel):
    """A drug, biological, device, or other intervention in a trial."""

    intervention_type: str = ""  # "Drug", "Biological", "Device", etc.
    intervention_name: str = ""  # e.g. "Semaglutide"
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class MeshTerm(BaseModel):
    """A MeSH term from ClinicalTrials.gov's derived conditionBrowseModule."""

    id: str = ""  # e.g. "D003924"
    term: str = ""  # e.g. "Diabetes Mellitus, Type 2"

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class PrimaryOutcome(BaseModel):
    """A primary outcome measure for a trial."""

    measure: str = ""  # what they're measuring
    time_frame: str | None = None  # e.g. "72 weeks"

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class Trial(BaseModel):
    """A single clinical trial record from ClinicalTrials.gov."""

    nct_id: str = ""
    title: str = ""
    brief_summary: str | None = None
    phase: str = ""  # "Phase 1", "Phase 2", "Phase 1/Phase 2", etc.
    overall_status: str = ""  # "Recruiting", "Completed", "Terminated", etc.
    why_stopped: str | None = None  # free text, only for Terminated/Withdrawn/Suspended
    indications: list[str] = []
    mesh_conditions: list[MeshTerm] = []
    mesh_ancestors: list[MeshTerm] = []
    interventions: list[Intervention] = []
    sponsor: str = ""
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    primary_outcomes: list[PrimaryOutcome] = []
    references: list[str] = []  # PMIDs

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


# ------------------------------------------------------------------
# Whitespace detection
# ------------------------------------------------------------------


class IndicationDrug(BaseModel):
    """A drug being tested for the same indication (when whitespace exists)."""

    nct_id: str = ""
    drug_name: str = ""
    indication: str = ""
    phase: str = ""
    status: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values

    @classmethod
    def from_trial(cls, trial: Trial, drug_name: str) -> "IndicationDrug":
        """Build an IndicationDrug from a Trial and its primary drug name."""
        return cls(
            nct_id=trial.nct_id,
            drug_name=drug_name,
            indication=trial.indications[0] if trial.indications else "",
            phase=trial.phase,
            status=trial.overall_status,
        )


class WhitespaceResult(BaseModel):
    """Result of whitespace detection — is this drug-indication pair unexplored?"""

    is_whitespace: bool | None = None
    no_data: bool | None = None
    exact_match_count: int | None = None
    drug_only_trials: int | None = None
    indication_only_trials: int | None = None
    indication_drugs: list[IndicationDrug] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


# ------------------------------------------------------------------
# Competitive landscape
# ------------------------------------------------------------------


class CompetitorEntry(BaseModel):
    """A sponsor + drug combination competing in a disease area."""

    sponsor: str = ""
    drug_name: str = ""
    drug_type: str | None = None
    max_phase: str = ""
    trial_count: int = 0
    statuses: set[str] = set()
    total_enrollment: int = 0
    most_recent_start: str | None = None  # ISO date of latest trial start

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class RecentStart(BaseModel):
    """A trial that started recently in an indication's landscape."""

    nct_id: str = ""
    sponsor: str = ""
    drug: str = ""
    phase: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class IndicationLandscape(BaseModel):
    """Full competitive landscape for an indication."""

    total_trial_count: int | None = None
    competitors: list[CompetitorEntry] = []
    phase_distribution: dict[str, int] = {}
    recent_starts: list[RecentStart] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


# ------------------------------------------------------------------
# Terminated trials (Critique Agent)
# ------------------------------------------------------------------


class TerminatedTrial(BaseModel):
    """A terminated, withdrawn, or suspended trial with stop classification."""

    nct_id: str = ""
    title: str = ""
    drug_name: str | None = None
    indication: str | None = None
    mesh_conditions: list[MeshTerm] = []
    phase: str | None = None
    why_stopped: str | None = None
    stop_category: str | None = (
        None  # safety, efficacy, business, enrollment, other, unknown
    )
    enrollment: int | None = None
    sponsor: str | None = None
    start_date: str | None = None
    termination_date: str | None = None
    references: list[str] = []  # PMIDs

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class TrialOutcomes(BaseModel):
    """Trial-outcome evidence split by scope, for repurposing analysis.

    Three termination scopes (trials stopped early):
    - drug_wide: this drug, any indication; safety/efficacy stop_categories only
      (the drug's overall failure history).
    - indication_wide: this indication, any drug (historical attrition in the
      disease area).
    - pair_specific: this drug AND this indication (the hypothesis has been
      directly tested and stopped); all stop_categories retained so the agent
      can distinguish efficacy/safety closures from business/enrollment ones.

    Plus one completed-trial scope (trials that ran to protocol end):
    - pair_completed: this drug AND this indication, status COMPLETED. Catches
      the common case where a Phase 3 trial finishes but misses its primary
      endpoint (ClinicalTrials.gov marks these COMPLETED, not TERMINATED).
      The agent should inspect these for likely outcome.
    """

    drug_wide: list[TerminatedTrial] = []
    indication_wide: list[TerminatedTrial] = []
    pair_specific: list[TerminatedTrial] = []
    pair_completed: list[Trial] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


# ------------------------------------------------------------------
# FDA approval check
# ------------------------------------------------------------------


class ApprovalCheck(BaseModel):
    """Result of an FDA-label lookup for a drug × indication pair.

    `is_approved` is True when the indication appears on a current
    FDA label for any known name of the drug. When False it means
    "not found on FDA labels" — it does not distinguish trial failure
    from approval pending from approval outside the US.

    `label_found` is True when FDA returned at least one label for any
    of the drug names checked. When False, no label exists in openFDA
    for this drug (e.g. withdrawn drugs like aducanumab after 2024) —
    approval status cannot be determined from available data.
    """

    is_approved: bool = False
    label_found: bool = False
    matched_indication: str | None = None
    drug_names_checked: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
