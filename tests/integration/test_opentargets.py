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
        """Test fetching aspirin returns a Drug model with expected fields."""
        result = await client.get_drug("CHEMBL25")

        assert result is not None
        assert isinstance(result, Drug)
        assert result.chembl_id == "CHEMBL25"
        assert result.name == "ASPIRIN"
        assert result.description is not None
        assert result.drug_type is not None

    async def test_get_drug_has_mechanisms(self, client: OpenTargetsClient):
        """Test that aspirin returns mechanism of action data."""
        result = await client.get_drug("CHEMBL25")

        assert len(result.mechanisms) > 0
        assert result.mechanisms[0].description is not None

    async def test_get_drug_has_indications(self, client: OpenTargetsClient):
        """Test that aspirin returns indication data."""
        result = await client.get_drug("CHEMBL25")

        assert len(result.indications) > 0
        assert result.indications[0].indication_id is not None
        assert result.indications[0].indication_name is not None

    async def test_get_drug_metformin(self, client: OpenTargetsClient):
        """Test fetching metformin - a common repurposing candidate."""
        result = await client.get_drug("CHEMBL1431")

        assert result is not None
        assert isinstance(result, Drug)
        assert result.chembl_id == "CHEMBL1431"
        assert "METFORMIN" in result.name.upper()

    async def test_get_drug_nonexistent_returns_none(self, client: OpenTargetsClient):
        """Test that a nonexistent ChEMBL ID returns None."""
        result = await client.get_drug("CHEMBL999999999")

        assert result is None
