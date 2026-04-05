"""Integration tests for clinical_trials_tools.

These tests hit real ClinicalTrials.gov APIs.
"""

import logging

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)

logger = logging.getLogger(__name__)

# Build tools once at module level (no date_before for integration tests)
_tools = build_clinical_trials_tools(date_before=None)
_tool_map = {t.name: t for t in _tools}

detect_whitespace = _tool_map["detect_whitespace"]
search_trials = _tool_map["search_trials"]
get_landscape = _tool_map["get_landscape"]
get_terminated = _tool_map["get_terminated"]


# ------------------------------------------------------------------
# detect_whitespace
# ------------------------------------------------------------------


async def test_detect_whitespace_whitespace():
    """detect_whitespace tool identifies unexplored drug-indication pair.

    Tirzepatide + Huntington disease = whitespace (no exact matches).
    Validates the tool returns a dict with correct structure and values,
    including indication_drugs ranked by phase.
    """
    result = await detect_whitespace.ainvoke(
        {"drug": "tirzepatide", "indication": "Huntington disease"}
    )

    assert result["is_whitespace"] is True
    assert result["exact_match_count"] == 0
    assert 150 < result["drug_only_trials"] < 300
    assert 200 < result["indication_only_trials"] < 400

    assert 40 < len(result["indication_drugs"]) < 60

    # Verify deduplication: all drug_names should be unique
    drug_names = [cd["drug_name"] for cd in result["indication_drugs"]]
    assert len(drug_names) == len(set(drug_names))

    # Known Phase 4 HD drugs must be present
    assert "Memantine" in drug_names
    assert "Tetrabenazine" in drug_names

    # Ranking: first entries should be Phase 4
    assert result["indication_drugs"][0]["phase"] == "Phase 4"
    assert result["indication_drugs"][1]["phase"] == "Phase 4"
    assert result["indication_drugs"][2]["phase"] == "Phase 4"

    # Verify all IndicationDrug fields on a known entry
    [memantine] = [
        cd for cd in result["indication_drugs"] if cd["drug_name"] == "Memantine"
    ]
    assert memantine["nct_id"] == "NCT00652457"
    assert memantine["drug_name"] == "Memantine"
    assert memantine["indication"] == "Huntington's Disease"
    assert memantine["phase"] == "Phase 4"
    assert memantine["status"] == "COMPLETED"

    # No Phase 1 trials (Phase 2+ filter)
    for cd in result["indication_drugs"]:
        assert "Phase 1" not in cd["phase"]
        assert "Early Phase 1" not in cd["phase"]


async def test_detect_whitespace_not_whitespace():
    """detect_whitespace tool returns is_whitespace=False when trials exist.

    Semaglutide + diabetes = NOT whitespace (many exact matches).
    indication_drugs should be empty.
    """
    result = await detect_whitespace.ainvoke(
        {"drug": "semaglutide", "indication": "diabetes"}
    )

    assert result["is_whitespace"] is False
    assert result["exact_match_count"] >= 10
    assert 400 < result["drug_only_trials"] < 800
    assert 10000 < result["indication_only_trials"] < 100000
    assert result["indication_drugs"] == []


# ------------------------------------------------------------------
# search_trials
# ------------------------------------------------------------------


async def test_search_trials_drug_and_indication():
    """search_trials tool returns trial dicts for a drug-indication pair.

    Results are sorted EnrollmentCount:desc. NCT03390608 is a Swedish
    nationwide registry study (n=35,002) and should consistently rank first.
    """
    result = await search_trials.ainvoke(
        {"drug": "trastuzumab", "indication": "breast cancer"}
    )

    assert len(result) >= 20

    # Highest-enrollment study should be present (n=35,002 Swedish registry)
    [top] = [t for t in result if t["nct_id"] == "NCT03390608"]
    assert top["nct_id"] == "NCT03390608"
    assert top["title"] == "Prognostic and Predictive Factors for Small Breast Tumors"
    assert top["phase"] == "Not Applicable"
    assert top["overall_status"] == "COMPLETED"
    assert top["why_stopped"] is None
    assert top["indications"] == ["Breast Cancer"]
    assert top["sponsor"] == "Karolinska Institutet"
    assert top["enrollment"] == 35002
    assert top["start_date"] == "1977-01-01"
    assert top["completion_date"] == "2014-12-31"
    assert top["references"] == []

    intervention_names = [i["intervention_name"] for i in top["interventions"]]
    assert "Herceptin" in intervention_names

    assert len(top["primary_outcomes"]) == 1
    assert top["primary_outcomes"][0]["measure"] == "Breast cancer specific death"
    assert (
        top["primary_outcomes"][0]["time_frame"]
        == "January 1, 1977 to December 31, 2014"
    )


async def test_search_trials_nonexistent_drug():
    """search_trials tool returns empty list for nonexistent drug."""
    result = await search_trials.ainvoke(
        {"drug": "xyzzy_not_a_real_drug_12345", "indication": "diabetes"}
    )

    assert result == []


# ------------------------------------------------------------------
# get_landscape
# ------------------------------------------------------------------


async def test_get_landscape_gastroparesis():
    """get_landscape tool returns competitive landscape dict."""
    result = await get_landscape.ainvoke({"indication": "gastroparesis"})

    # Gastroparesis has ~300+ trials
    assert result["total_trial_count"] > 200

    # Tool passes top_n=10
    assert len(result["competitors"]) == 10

    # Phase distribution
    assert 10 < result["phase_distribution"]["Phase 2"] < 100
    assert 5 < result["phase_distribution"]["Phase 3"] < 50
    assert 1 < result["phase_distribution"]["Phase 4"] < 30

    # Known recent start
    assert len(result["recent_starts"]) >= 1
    [tradipitant] = [
        rs for rs in result["recent_starts"] if rs["nct_id"] == "NCT06836557"
    ]
    assert tradipitant["sponsor"] == "Vanda Pharmaceuticals"
    assert tradipitant["drug"] == "Tradipitant"
    assert tradipitant["phase"] == "Phase 3"

    # Known competitor - verify all fields
    [cuhk] = [
        c
        for c in result["competitors"]
        if c["sponsor"] == "Chinese University of Hong Kong"
        and c["drug_name"] == "Esomeprazole"
    ]
    assert cuhk["sponsor"] == "Chinese University of Hong Kong"
    assert cuhk["drug_name"] == "Esomeprazole"
    assert cuhk["drug_type"] == "Drug"
    assert cuhk["max_phase"] == "Phase 4"
    assert cuhk["trial_count"] == 1
    assert cuhk["total_enrollment"] == 155


async def test_get_landscape_nonexistent_indication():
    """get_landscape tool returns empty landscape for nonexistent indication."""
    result = await get_landscape.ainvoke({"indication": "xyzzy_fake_indication_99999"})

    assert result["total_trial_count"] == 0
    assert result["competitors"] == []
    assert result["phase_distribution"] == {}


# ------------------------------------------------------------------
# get_terminated
# ------------------------------------------------------------------


async def test_get_terminated_semaglutide():
    """get_terminated tool returns terminated trial dicts."""
    result = await get_terminated.ainvoke(
        {"drug": "troglitazone", "indication": "diabetes"}
    )

    assert len(result) >= 1
    assert any(t["stop_category"] == "safety" for t in result)


async def test_get_terminated_nonexistent_query():
    """get_terminated tool returns empty list for nonexistent drug and indication."""
    result = await get_terminated.ainvoke(
        {
            "drug": "xyzzy_not_a_real_term_12345",
            "indication": "xyzzy_not_a_real_indication_12345",
        }
    )

    assert result == []
