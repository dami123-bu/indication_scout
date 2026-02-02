"""DrugBank data client."""


class DrugBankClient:
    """Client for accessing DrugBank data."""

    async def get_drug(self, drug_name: str) -> dict | None:
        """Get drug information by name."""
        raise NotImplementedError

    async def get_interactions(self, drug_id: str) -> list[dict]:
        """Get drug-drug interactions."""
        raise NotImplementedError
