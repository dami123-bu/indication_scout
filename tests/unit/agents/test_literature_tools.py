"""Unit tests for literature_tools."""

import logging
from unittest.mock import AsyncMock, MagicMock

from indication_scout.agents.literature.literature_tools import build_literature_tools
from indication_scout.models.model_drug_profile import DrugProfile

logger = logging.getLogger(__name__)


def _drug_profile() -> DrugProfile:
    return DrugProfile(
        name="metformin",
        synonyms=["Glucophage"],
        target_gene_symbols=["PRKAA1"],
        mechanisms_of_action=["AMP-activated protein kinase activator"],
        atc_codes=["A10BA02"],
        atc_descriptions=["Biguanides"],
        drug_type="Small molecule",
    )


def _get_tool(tools: list, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found in {[t.name for t in tools]}")


async def test_expand_search_terms_returns_queries():
    """expand_search_terms passes drug_profile from closure and returns queries from svc."""
    expected_queries = ["metformin colorectal cancer", "AMPK colon neoplasm"]
    svc = MagicMock()
    svc.expand_search_terms = AsyncMock(return_value=expected_queries)
    profile = _drug_profile()

    tools = build_literature_tools(svc=svc, db=MagicMock(), drug_profile=profile)
    expand = _get_tool(tools, "expand_search_terms")

    result = await expand.ainvoke(
        {"drug_name": "metformin", "disease_name": "colorectal cancer"}
    )

    svc.expand_search_terms.assert_awaited_once_with(
        "metformin", "colorectal cancer", profile
    )
    assert result == ["metformin colorectal cancer", "AMPK colon neoplasm"]
