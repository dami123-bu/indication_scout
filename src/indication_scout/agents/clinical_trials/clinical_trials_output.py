from pydantic import BaseModel, Field

from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    WhitespaceResult,
    Trial,
    IndicationLandscape,
    TrialOutcomes,
)


class ClinicalTrialsOutput(BaseModel):
    """Final assembled output from a single clinical trials agent run."""

    whitespace: WhitespaceResult | None = Field(
        default=None,
        description=(
            "Whitespace analysis: trial counts and competing drugs for the indication."
        ),
    )

    landscape: IndicationLandscape | None = Field(
        default=None,
        description=("Competitive landscape for the indication."),
    )

    trials: list[Trial] = Field(
        default_factory=list,
        description=("Trials matching the drug-indication pair."),
    )

    terminated: TrialOutcomes = Field(
        default_factory=TrialOutcomes,
        description=(
            "Trial-outcome evidence split by scope: drug_wide, indication_wide, "
            "pair_specific (terminated), pair_completed."
        ),
    )

    approval: ApprovalCheck | None = Field(
        default=None,
        description=(
            "FDA-label approval status for the drug × indication pair. Populated "
            "when the agent calls check_fda_approval."
        ),
    )

    summary: str = Field(
        default="",
        description="LLM narrative summary from the final agent message.",
    )
