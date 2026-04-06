from datetime import date
from typing import Annotated, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.models.model_evidence_summary import EvidenceSummary


class LiteratureState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Current context
    drug_name: str = ""
    disease_name: str = ""

    @field_validator("drug_name")
    @classmethod
    def lowercase_drug_name(cls, v: str) -> str:
        return v.lower()

    date_before: Optional[date] = None

    expanded_search_results: list[str] = []
    pmids: list[str] = []
    summary: str = ""
    semantic_search_results: list[dict] = []
    evidence_summary: EvidenceSummary | None = None

    # Final assembled output
    final_output: Optional[LiteratureOutput] = None

    # Required because we store custom Pydantic models inside the state
    model_config = {"arbitrary_types_allowed": True}
