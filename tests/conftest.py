"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_drug() -> dict:
    """Sample drug data for testing."""
    return {
        "id": "DB00316",
        "name": "Acetaminophen",
        "synonyms": ["Tylenol", "Paracetamol"],
    }


@pytest.fixture
def sample_indication() -> dict:
    """Sample indication data for testing."""
    return {
        "id": "C0030193",
        "name": "Pain",
        "synonyms": ["Chronic Pain", "Acute Pain"],
    }
