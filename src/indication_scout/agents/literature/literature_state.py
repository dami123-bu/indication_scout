"""State for the literature analysis LangGraph agent."""

from datetime import date
from typing import Annotated, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import AbstractResult


class LiteratureState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Context
    drug_name: str = ""
    disease_name: str = ""
    date_before: Optional[date] = None

    # Tool results — populated via msg.artifact in tools_node
    drug_profile: Optional[DrugProfile] = None
    expanded_search_results: list[str] = Field(default_factory=list)
    pmids: list[str] = Field(default_factory=list)
    semantic_search_results: list[AbstractResult] = Field(default_factory=list)
    evidence_summary: Optional[EvidenceSummary] = None

    @field_validator(
        "expanded_search_results", "pmids", "semantic_search_results", mode="before"
    )
    @classmethod
    def none_to_list(cls, v):
        return v if v is not None else []

    # Final assembled output
    final_output: Optional[LiteratureOutput] = None

    model_config = {"arbitrary_types_allowed": True}
