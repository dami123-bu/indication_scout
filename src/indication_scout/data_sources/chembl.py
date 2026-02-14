"""ChEMBL API client."""

from indication_scout.data_sources.base_client import BaseClient


class ChEMBLClient(BaseClient):
    """Client for querying ChEMBL database."""

    @property
    def _source_name(self) -> str:
        return "chembl"

    async def search_compound(self, name: str) -> dict | None:
        """Search for a compound by name."""
        raise NotImplementedError

    async def get_activities(self, chembl_id: str) -> list[dict]:
        """Get bioactivity data for a compound."""
        raise NotImplementedError
