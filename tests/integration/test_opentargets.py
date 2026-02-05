"""Integration tests for OpenTargets client."""

import pytest

from indication_scout.data.opentargets import OpenTargetsClient
from indication_scout.models.drug import Drug

pytestmark = pytest.mark.asyncio


@pytest.fixture
def client() -> OpenTargetsClient:
    """Create OpenTargets client instance."""
    return OpenTargetsClient()


class TestResolveId:
    """Integration tests for resolve_id method."""

    async def test_resolve_drug_aspirin(self, client: OpenTargetsClient):
        """Test resolving a well-known drug returns a ChEMBL ID."""
        result = await client.resolve_id("aspirin", "drug")

        assert result is not None
        assert result.startswith("CHEMBL")

    async def test_resolve_disease_multiple_sclerosis(self, client: OpenTargetsClient):
        """Test resolving a well-known disease returns an EFO ID."""
        result = await client.resolve_id("multiple sclerosis", "disease")

        assert result is not None
        # Open Targets disease IDs are typically EFO, MONDO, or Orphanet
        assert any(
            result.startswith(prefix) for prefix in ["EFO_", "MONDO_", "Orphanet_"]
        )

    async def test_resolve_target_egfr(self, client: OpenTargetsClient):
        """Test resolving a well-known target returns an Ensembl ID."""
        result = await client.resolve_id("EGFR", "target")

        assert result is not None
        assert result.startswith("ENSG")

    async def test_resolve_nonexistent_returns_none(self, client: OpenTargetsClient):
        """Test that a nonsense query returns None."""
        result = await client.resolve_id("xyzzy12345notarealthing", "drug")

        assert result is None

    async def test_resolve_drug_metformin(self, client: OpenTargetsClient):
        """Test resolving metformin - a common repurposing candidate."""
        result = await client.resolve_id("metformin", "drug")

        assert result is not None
        assert result.startswith("CHEMBL")


class TestGetDrug:
    """Integration tests for get_drug method."""

    async def test_get_drug_aspirin(self, client: OpenTargetsClient):
        """Test that get_drug parses scalar fields for a well-known drug."""
        result = await client.get_drug("CHEMBL25")

        assert isinstance(result, Drug)
        assert result.chembl_id == "CHEMBL25"
        assert result.generic_name == "ASPIRIN"
        assert result.description is not None
        assert result.drug_type is not None
        assert result.is_approved is True
        assert result.has_been_withdrawn is False
        assert result.max_clinical_phase is not None
        assert len(result.synonyms) > 0

    async def test_get_drug_has_activities(self, client: OpenTargetsClient):
        """Test that aspirin has DrugActivity entries with targets."""
        result = await client.get_drug("CHEMBL25")

        assert len(result.activities) > 0
        assert result.activities[0].description is not None
        # At least one activity should have a target
        targets = [a.target for a in result.activities if a.target is not None]
        assert len(targets) > 0
        assert targets[0].ensembl_id is not None
        assert targets[0].symbol is not None

    async def test_get_drug_nonexistent_returns_none(self, client: OpenTargetsClient):
        """Test that a nonexistent ChEMBL ID returns None."""
        result = await client.get_drug("CHEMBL999999999")

        assert result is None
