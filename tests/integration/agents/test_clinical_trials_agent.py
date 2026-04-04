"""Integration tests for build_clinical_trials_graph.

These tests hit real ClinicalTrials.gov and Anthropic APIs.
They verify the agent calls the right tools and produces structured output.
"""

import logging

from langchain_core.messages import HumanMessage

from indication_scout.constants import CLINICAL_TRIALS_RECURSION_LIMIT

logger = logging.getLogger(__name__)


async def _run(graph, drug: str, disease: str):
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=f"Analyze {drug} in {disease}")],
            "drug_name": drug,
            "disease_name": disease,
            "date_before": None,
        },
        config={"recursion_limit": CLINICAL_TRIALS_RECURSION_LIMIT},
    )
    return result["final_output"]


# ------------------------------------------------------------------
# Whitespace path: drug-disease pair with no trials
# ------------------------------------------------------------------


async def test_agent_whitespace_path(clinical_trials_graph):
    """Agent detects whitespace for tirzepatide + Huntington disease.

    Expected behavior: detect_whitespace → whitespace found →
    get_terminated → get_landscape → summary.
    """
    output = await _run(clinical_trials_graph, "tirzepatide", "Huntington disease")

    # Whitespace detected
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is True
    assert output.whitespace.exact_match_count == 0
    assert output.whitespace.drug_only_trials > 100
    assert output.whitespace.indication_only_trials > 100

    # Agent should have populated indication_drugs
    assert len(output.whitespace.indication_drugs) > 10
    drug_names = [d.drug_name for d in output.whitespace.indication_drugs]
    assert "Memantine" in drug_names
    assert "Tetrabenazine" in drug_names

    # No exact-match trials for this pair
    assert output.trials == []

    # Agent should have produced a summary
    assert len(output.summary) > 50


# ------------------------------------------------------------------
# Active trials path: drug-disease pair with trials
# ------------------------------------------------------------------


async def test_agent_active_trials_path(clinical_trials_graph):
    """Agent finds active trials for tofacitinib + alopecia areata.

    Expected behavior: detect_whitespace → not whitespace →
    search_trials → get_landscape → summary.
    """
    output = await _run(clinical_trials_graph, "tofacitinib", "alopecia areata")

    # Not whitespace
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is False
    assert output.whitespace.exact_match_count >= 5

    # Agent should have fetched trial details
    assert len(output.trials) >= 5
    for trial in output.trials:
        assert trial.nct_id.startswith("NCT")

    # Agent should have produced a summary
    assert len(output.summary) > 50


# ------------------------------------------------------------------
# Nonexistent drug: no data at all
# ------------------------------------------------------------------


async def test_agent_no_data(clinical_trials_graph):
    """Agent handles nonexistent drug gracefully.

    Expected behavior: detect_whitespace → whitespace + no_data →
    brief summary noting lack of data.
    """
    output = await _run(clinical_trials_graph, "xyzzy_fake_drug_99999", "xyzzy_fake_disease_99999")

    # Whitespace with no data
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is True
    assert output.whitespace.exact_match_count == 0
    assert output.whitespace.drug_only_trials == 0
    assert output.whitespace.indication_only_trials == 0

    # No trials
    assert output.trials == []

    # Agent should still produce a summary
    assert len(output.summary) > 20
