"""ChEMBL API client."""


class ChEMBLClient:
    """Client for querying ChEMBL database."""

    async def search_compound(self, name: str) -> dict | None:
        """Search for a compound by name."""
        raise NotImplementedError

    async def get_activities(self, chembl_id: str) -> list[dict]:
        """Get bioactivity data for a compound."""
        raise NotImplementedError
