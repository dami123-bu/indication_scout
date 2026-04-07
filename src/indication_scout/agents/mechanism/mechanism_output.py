"""Structured output from the mechanism agent."""

from pydantic import BaseModel, Field

from indication_scout.models.model_open_targets import Association, Pathway


class MechanismOutput(BaseModel):
    """Final assembled output from a single mechanism agent run."""

    drug_targets: dict[str, str] = Field(
        default_factory=dict,
        description="Gene symbol → Ensembl target ID for each of the drug's targets",
    )
    associations: dict[str, list[Association]] = Field(
        default_factory=dict,
        description="Top disease associations per target (symbol → associations)",
    )
    pathways: dict[str, list[Pathway]] = Field(
        default_factory=dict,
        description="Reactome pathways per target (symbol → pathways)",
    )
    summary: str = Field(default="", description="Narrative summary from the agent")
