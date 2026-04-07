"""Structured output from the mechanism  agent."""

from pydantic import BaseModel, Field

from indication_scout.models.model_drug_profile import DrugProfile


class MechanismOutput(BaseModel):
    """Final assembled output from a single mechanism agent run."""

    competitors: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Competitor drugs grouped by disease (disease → list of drug names)",
    )
    drug_indications: list[str] = Field(
        default_factory=list,
        description="Approved indications for this drug",
    )
    drug_profile: DrugProfile | None = Field(
        default=None,
        description="Pharmacological profile of the drug being analyzed",
    )
    search_queries: dict[str, list[str]] = Field(
        default_factory=dict,
        description="PubMed queries per disease (disease → list of query strings)",
    )
    summary: str = Field(default="", description="Narrative summary from the agent")

