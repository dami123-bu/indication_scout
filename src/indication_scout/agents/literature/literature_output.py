"""Output model for the LiteratureOutput."""

import datetime

from pydantic import BaseModel, Field

from indication_scout.models.model_evidence_summary import EvidenceSummary


class LiteratureOutput(BaseModel):
    evidence_summary: EvidenceSummary | None = None
    search_results: list[str] = []
    pmids: list[str] = []
    semantic_search_results: list[dict] = []

    summary: str = ""  # 2-3 sentence natural language assessment from the agent

    # Optional metadata
    analyzed_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
