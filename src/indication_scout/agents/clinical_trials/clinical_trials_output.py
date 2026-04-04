"""Output model for the ClinicalTrialsAgent."""
import datetime

from pydantic import BaseModel, model_validator, Field

from indication_scout.models.model_clinical_trials import (
    IndicationLandscape,
    TerminatedTrial,
    Trial,
    WhitespaceResult,
)


class ClinicalTrialsOutput(BaseModel):
    """Structured output from the ClinicalTrialsAgent.

    Fields are None/empty when the agent chose not to call the corresponding
    tool — this is expected behavior, not an error.
    """

    trials: list[Trial] = Field(default_factory=list)
    whitespace: WhitespaceResult | None = None
    landscape: IndicationLandscape | None = None
    terminated: list[TerminatedTrial] = Field(default_factory=list)
    summary: str = ""  # 2-3 sentence natural language assessment from the agent

    # Optional metadata
    analyzed_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
