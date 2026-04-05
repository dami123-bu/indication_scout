"""Integration tests for build_literature_graph.

These tests hit real Anthropic and Open Targets APIs.
They verify the agent calls expand_search_terms and produces structured output.
"""

import logging
from datetime import date

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from indication_scout.agents.literature.literature_agent import build_literature_graph
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 10


async def _run(drug: str, disease: str, date_before: date | None = None) -> object:
    svc = RetrievalService(DEFAULT_CACHE_DIR)
    drug_profile = await svc.build_drug_profile(drug)
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    graph = build_literature_graph(llm=llm, svc=svc, drug_profile=drug_profile, max_search_results=20)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=f"Analyze {drug} in {disease}")],
            "drug_name": drug,
            "disease_name": disease,
            "date_before": date_before,
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return result["final_output"]


# ------------------------------------------------------------------
# bupropion / depression — known drug-disease pair
# ------------------------------------------------------------------


async def test_literature_agent_bupropion_depression():
    """Agent generates search terms for bupropion in depression.

    Expected: expand_search_terms produces 8 queries derived from the bupropion
    drug profile (synonyms, targets, MOA, ATC), and a substantive summary.
    """
    output = await _run("bupropion", "depression")

    assert output.search_results == [
        "bupropion AND depression",
        "antidepressants AND brain",
        "dopamine transporter inhibitor AND brain",
        "norepinephrine transporter inhibitor AND depression",
        "SLC6A3 AND depression",
        "SLC6A2 AND depression",
        "amfebutamone AND depression",
        "bupropion extended release AND major depressive disorder",
    ]

    assert len(output.summary) > 100
