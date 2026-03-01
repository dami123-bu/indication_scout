"""Shared fixtures for integration tests."""

import pytest

from indication_scout.constants import TEST_CACHE_DIR
from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.data_sources.pubmed import PubMedClient


@pytest.fixture
async def chembl_client():
    """Create and tear down a ChEMBLClient."""
    c = ChEMBLClient()
    yield c
    await c.close()


@pytest.fixture
async def open_targets_client():
    """Create and tear down an OpenTargetsClient using the test cache."""
    c = OpenTargetsClient(cache_dir=TEST_CACHE_DIR)
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
