from datetime import date
from typing import Annotated, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from indication_scout.agents.literature.literature_output import LiteratureOutput


class LiteratureState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Current context
    drug_name: str = ""
    disease_name: str = ""
    date_before: Optional[date] = None

    search_results: list[str] = []
    pmids: list[str] = []
    summary: str = ""

    # Final assembled output
    final_output: Optional[LiteratureOutput] = None

    # Required because we store custom Pydantic models inside the state
    model_config = {"arbitrary_types_allowed": True}
