"""Structured output from the mechanism agent."""

from pydantic import BaseModel, Field

from indication_scout.models.model_open_targets import MechanismOfAction


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
    summary: str = Field(default="", description="Narrative summary from the agent")
