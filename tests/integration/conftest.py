"""Shared fixtures for integration tests."""

import os

import pytest
from dotenv import load_dotenv

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.data_sources.fda import FDAClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.data_sources.pubmed import PubMedClient

load_dotenv()


@pytest.fixture
async def open_targets_client():
    """Create and tear down an OpenTargetsClient."""
    c = OpenTargetsClient()
    yield c
    await c.close()


@pytest.fixture
async def pubmed_client():
    """Create and tear down a PubMedClient."""
    c = PubMedClient()
    yield c
    await c.close()


@pytest.fixture
async def clinical_trials_client():
    """Create and tear down a ClinicalTrialsClient."""
    c = ClinicalTrialsClient()
    yield c
    await c.close()


@pytest.fixture
async def fda_client():
    """Create and tear down an FDAClient."""
    api_key = os.environ.get("OPENFDA_API_KEY", "")
    c = FDAClient(api_key=api_key)
    yield c
    await c.close()
