"""ClinicalTrials.gov API client."""


class ClinicalTrialsClient:
    """Client for querying ClinicalTrials.gov API."""

    async def search(self, drug: str, condition: str | None = None) -> list[dict]:
        """Search for clinical trials involving a drug."""
        raise NotImplementedError

    async def get_trial(self, nct_id: str) -> dict:
        """Fetch details for a specific trial by NCT ID."""
        raise NotImplementedError
