"""Unit tests for FDAClient."""

import pytest

from indication_scout.constants import REACTION_OUTCOME_MAP
from indication_scout.data_sources.fda import FDAClient


# --- _parse_reaction_count ---


def test_parse_reaction_count():
    """Test parsing a valid reaction count result."""
    raw = {"term": "Nausea", "count": 1234}

    result = FDAClient._parse_reaction_count(raw)

    assert result.term == "Nausea"
    assert result.count == 1234


def test_parse_reaction_count_zero_count():
    """Test parsing a reaction count with zero reports."""
    raw = {"term": "Headache", "count": 0}

    result = FDAClient._parse_reaction_count(raw)

    assert result.term == "Headache"
    assert result.count == 0


# --- _parse_event ---


def test_parse_event_all_fields():
    """Test parsing a full event with all fields populated."""
    raw = {
        "patient": {
            "drug": [
                {
                    "medicinalproduct": "METFORMIN",
                    "drugindication": "Type 2 diabetes mellitus",
                }
            ],
            "reaction": [
                {
                    "reactionmeddrapt": "Lactic acidosis",
                    "reactionoutcome": "1",
                }
            ],
        },
        "serious": "1",
        "companynumb": "US-FDA-2024-001",
    }

    result = FDAClient._parse_event(raw)

    assert result.medicinal_product == "METFORMIN"
    assert result.drug_indication == "Type 2 diabetes mellitus"
    assert result.reaction == "Lactic acidosis"
    assert result.reaction_outcome == "Recovered/Resolved"
    assert result.serious == "1"
    assert result.company_numb == "US-FDA-2024-001"


def test_parse_event_missing_optional_fields():
    """Test parsing an event where optional fields are absent."""
    raw = {
        "patient": {
            "drug": [
                {
                    "medicinalproduct": "METFORMIN",
                }
            ],
            "reaction": [
                {
                    "reactionmeddrapt": "Nausea",
                }
            ],
        },
    }

    result = FDAClient._parse_event(raw)

    assert result.medicinal_product == "METFORMIN"
    assert result.drug_indication is None
    assert result.reaction == "Nausea"
    assert result.reaction_outcome is None
    assert result.serious is None
    assert result.company_numb is None


def test_parse_event_empty_drug_list():
    """Test parsing an event with no drugs in the patient record."""
    raw = {
        "patient": {
            "drug": [],
            "reaction": [
                {
                    "reactionmeddrapt": "Fatigue",
                    "reactionoutcome": "1",
                }
            ],
        },
    }

    result = FDAClient._parse_event(raw)

    assert result.medicinal_product == ""
    assert result.drug_indication is None
    assert result.reaction == "Fatigue"
    assert result.reaction_outcome == "Recovered/Resolved"


def test_parse_event_empty_reaction_list():
    """Test parsing an event with no reactions in the patient record."""
    raw = {
        "patient": {
            "drug": [
                {
                    "medicinalproduct": "METFORMIN",
                    "drugindication": "Type 2 diabetes mellitus",
                }
            ],
            "reaction": [],
        },
    }

    result = FDAClient._parse_event(raw)

    assert result.medicinal_product == "METFORMIN"
    assert result.drug_indication == "Type 2 diabetes mellitus"
    assert result.reaction == ""
    assert result.reaction_outcome is None


@pytest.mark.parametrize(
    "code,expected",
    [
        ("1", "Recovered/Resolved"),
        ("2", "Not Recovered/Not Resolved"),
        ("3", "Recovering/Resolving"),
        ("4", "Recovered/Resolved with Sequelae"),
        ("5", "Fatal"),
        ("6", "Unknown"),
    ],
)
def test_parse_event_outcome_mapping(code: str, expected: str):
    """Test that each outcome code maps to the correct human-readable string."""
    raw = {
        "patient": {
            "drug": [{"medicinalproduct": "METFORMIN"}],
            "reaction": [
                {
                    "reactionmeddrapt": "Nausea",
                    "reactionoutcome": code,
                }
            ],
        },
    }

    result = FDAClient._parse_event(raw)

    assert result.reaction_outcome == expected


def test_parse_event_unknown_outcome_code():
    """Test that an unrecognized outcome code maps to None."""
    raw = {
        "patient": {
            "drug": [{"medicinalproduct": "METFORMIN"}],
            "reaction": [
                {
                    "reactionmeddrapt": "Nausea",
                    "reactionoutcome": "99",
                }
            ],
        },
    }

    result = FDAClient._parse_event(raw)

    assert result.reaction_outcome is None


# --- _build_params ---


def test_build_params_without_api_key():
    """Test param building when no API key is set."""
    client = FDAClient(api_key="")

    params = client._build_params("metformin", 10)

    assert params["search"] == 'patient.drug.medicinalproduct:"metformin"'
    assert params["limit"] == "10"
    assert "api_key" not in params


def test_build_params_with_api_key():
    """Test param building when API key is provided."""
    client = FDAClient(api_key="test_key_123")

    params = client._build_params("metformin", 10)

    assert params["search"] == 'patient.drug.medicinalproduct:"metformin"'
    assert params["limit"] == "10"
    assert params["api_key"] == "test_key_123"


def test_build_params_limit_capped():
    """Test that limit is capped at OPENFDA_MAX_LIMIT (1000)."""
    client = FDAClient(api_key="")

    params = client._build_params("metformin", 5000)

    assert params["limit"] == "1000"
