"""Unit tests for OpenTargetsClient."""

from unittest.mock import AsyncMock, patch

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import (
    DrugData,
    DrugSummary,
    DrugTarget,
    Indication,
)

# --- OpenTargetsClient configuration ---


def test_default_config():
    """Test that client uses default settings."""
    client = OpenTargetsClient()

    assert client.timeout == 30.0
    assert client.max_retries == 3
    assert client.cache_dir is not None
    assert client.cache_dir == DEFAULT_CACHE_DIR


# --- get_drug_competitors: None phase guard ---


async def test_get_drug_competitors_skips_summary_with_none_phase(tmp_path):
    """Summaries with phase=None must be skipped without raising TypeError."""
    drug = DrugData(
        chembl_id="CHEMBL1",
        name="testdrug",
        targets=[DrugTarget(target_id="ENSG001", target_symbol="TGT1")],
        indications=[],
    )
    summaries = [
        DrugSummary(drug_name="competitor_a", disease_name="depression", phase=3),
        DrugSummary(drug_name="competitor_b", disease_name="anxiety", phase=None),
    ]

    client = OpenTargetsClient(cache_dir=tmp_path)
    with (
        patch.object(client, "get_drug", new=AsyncMock(return_value=drug)),
        patch.object(
            client,
            "get_target_data_drug_summaries",
            new=AsyncMock(return_value=summaries),
        ),
    ):
        result = await client.get_drug_competitors("testdrug", drug_phase=3)

    # anxiety (phase=None) must be absent; depression (phase=3) must be present
    assert "anxiety" not in result["diseases"]
    assert "depression" in result["diseases"]


# --- get_drug_competitors: raw diseases returned ---


async def test_get_drug_competitors_returns_raw_diseases(tmp_path):
    """get_drug_competitors returns CompetitorRawData with unmerged diseases and indications."""
    drug = DrugData(
        chembl_id="CHEMBL1",
        name="testdrug",
        targets=[DrugTarget(target_id="ENSG001", target_symbol="TGT1")],
        indications=[],
    )
    summaries = [
        DrugSummary(drug_name="competitor_a", disease_name="narcolepsy", phase=3),
        DrugSummary(
            drug_name="competitor_b",
            disease_name="narcolepsy-cataplexy syndrome",
            phase=3,
        ),
    ]

    client = OpenTargetsClient(cache_dir=tmp_path)
    with (
        patch.object(client, "get_drug", new=AsyncMock(return_value=drug)),
        patch.object(
            client,
            "get_target_data_drug_summaries",
            new=AsyncMock(return_value=summaries),
        ),
    ):
        result = await client.get_drug_competitors("testdrug", drug_phase=3)

    assert "narcolepsy" in result["diseases"]
    assert "narcolepsy-cataplexy syndrome" in result["diseases"]
    assert result["drug_indications"] == []
