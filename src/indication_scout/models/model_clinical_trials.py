"""
Pydantic models for ClinicalTrials.gov data.

These are the data contracts between the ClinicalTrials.gov client and the agents.
Agents receive these models — they never see raw API responses.
"""

from pydantic import BaseModel

# ------------------------------------------------------------------
# Trial-level models
# ------------------------------------------------------------------


class Intervention(BaseModel):
    """A drug, biological, device, or other intervention in a trial."""

    intervention_type: str  # "Drug", "Biological", "Device", etc.
    intervention_name: str  # e.g. "Semaglutide"
    description: str | None = None


class PrimaryOutcome(BaseModel):
    """A primary outcome measure for a trial."""

    measure: str  # what they're measuring
    time_frame: str | None = None  # e.g. "72 weeks"


class Trial(BaseModel):
    """A single clinical trial record from ClinicalTrials.gov."""

    nct_id: str
    title: str
    brief_summary: str | None = None
    phase: str  # "Phase 1", "Phase 2", "Phase 1/Phase 2", etc.
    overall_status: str  # "Recruiting", "Completed", "Terminated", etc.
    why_stopped: str | None = None  # free text, only for Terminated/Withdrawn/Suspended
    conditions: list[str] = []
    interventions: list[Intervention] = []
    sponsor: str = ""
    collaborators: list[str] = []
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    study_type: str = "Interventional"
    primary_outcomes: list[PrimaryOutcome] = []
    results_posted: bool = False
    references: list[str] = []  # PMIDs


# ------------------------------------------------------------------
# Whitespace detection
# ------------------------------------------------------------------


class NearMiss(BaseModel):
    """A trial that's close but not an exact match for the drug-condition pair."""

    nct_id: str
    drug_name: str
    condition: str
    phase: str
    status: str


class WhitespaceResult(BaseModel):
    """Result of whitespace detection — is this drug-condition pair unexplored?"""

    is_whitespace: bool
    exact_match_count: int
    drug_only_trials: int
    condition_only_trials: int
    near_misses: list[NearMiss] = []


# ------------------------------------------------------------------
# Competitive landscape
# ------------------------------------------------------------------


class CompetitorEntry(BaseModel):
    """A sponsor + drug combination competing in a disease area."""

    sponsor: str
    drug_name: str
    drug_type: str | None = None
    max_phase: str
    trial_count: int
    statuses: set[str]
    total_enrollment: int = 0
    most_recent_start: str | None = None


class ConditionLandscape(BaseModel):
    """Full competitive landscape for a condition."""

    total_trial_count: int
    competitors: list[CompetitorEntry] = []
    phase_distribution: dict[str, int] = {}
    recent_starts: list[dict] = []


# ------------------------------------------------------------------
# Terminated trials (Critique Agent)
# ------------------------------------------------------------------


class TerminatedTrial(BaseModel):
    """A terminated, withdrawn, or suspended trial with stop classification."""

    nct_id: str
    title: str
    drug_name: str | None = None
    condition: str | None = None
    phase: str | None = None
    why_stopped: str | None = None
    stop_category: str = (
        "unknown"  # safety, efficacy, business, enrollment, other, unknown
    )
    enrollment: int | None = None
    sponsor: str | None = None
    start_date: str | None = None
    termination_date: str | None = None
    references: list[str] = []  # PMIDs
