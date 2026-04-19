"""Structured output from the mechanism agent."""

from typing import Literal

from pydantic import BaseModel, Field

from indication_scout.models.model_open_targets import MechanismOfAction, Pathway


class ShapedAssociation(BaseModel):
    """A disease association annotated with mechanistic directionality.

    This is the single per-(target, disease) record carried on MechanismOutput.
    The underlying Open Targets Association carries extra fields — datatype_scores
    (per-evidence-type scores) and therapeutic_areas — which are consumed during
    shape computation and baked into `rationale`, but are NOT propagated here.
    Downstream consumers read only the fields on this model.
    """

    target_symbol: str = Field(description="Gene symbol of the target")
    disease_name: str = Field(description="Disease name from Open Targets")
    disease_id: str = Field(description="Disease ID from Open Targets")
    overall_score: float | None = Field(
        default=None,
        description="Open Targets overall association score — used for threshold filtering",
    )
    shape: Literal["hypothesis", "contraindication", "neutral", "confirms_known"] = Field(
        description=(
            "hypothesis: drug action direction matches disease mechanism; "
            "contraindication: drug action opposes disease mechanism; "
            "neutral: insufficient evidence to determine direction; "
            "confirms_known: association is dominated by the drug's own clinical use"
        )
    )
    rationale: str = Field(
        description="One sentence explaining why this shape was assigned, referencing the action type and disease mechanism"
    )


class MechanismOutput(BaseModel):
    """Final assembled output from a single mechanism agent run."""

    drug_targets: dict[str, str] = Field(
        default_factory=dict,
        description="Gene symbol → Ensembl target ID for each of the drug's targets",
    )
    mechanisms_of_action: list[MechanismOfAction] = Field(
        default_factory=list,
        description="Structured MoA entries from the drug entity: action type, mechanism string, and targets",
    )
    shaped_associations: list[ShapedAssociation] = Field(
        default_factory=list,
        description="Per-association directionality judgements from the agent",
    )
    pathways: dict[str, list[Pathway]] = Field(
        default_factory=dict,
        description="Reactome pathways per target (symbol → pathways)",
    )
    summary: str = Field(default="", description="Narrative summary from the agent")
