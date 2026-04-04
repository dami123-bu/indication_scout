from datetime import date
from typing import Annotated, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.models.model_clinical_trials import (
    Trial,
    WhitespaceResult,
    IndicationLandscape,
    TerminatedTrial,
)


class ClinicalTrialsState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Current context
    drug_name: str = ""
    disease_name: str = ""
    date_before: Optional[date] = None

    # Structured results from tools
    trials: list[Trial] = Field(default_factory=list)
    whitespace: Optional[WhitespaceResult] = None
    landscape: Optional[IndicationLandscape] = None
    terminated: list[TerminatedTrial] = Field(default_factory=list)

    # Final assembled output
    final_output: Optional[ClinicalTrialsOutput] = None

    # Required because we store custom Pydantic models inside the state
    model_config = {"arbitrary_types_allowed": True}
