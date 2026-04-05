"""Output model for the LiteratureOutput."""

import datetime

from pydantic import BaseModel, Field


class LiteratureOutput(BaseModel):
    # evidence_summary:EvidenceSummary
    search_results: list[str]

    summary: str = ""  # 2-3 sentence natural language assessment from the agent

    # Optional metadata
    analyzed_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
