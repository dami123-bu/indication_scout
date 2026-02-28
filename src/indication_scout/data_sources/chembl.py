"""ChEMBL API client."""

import logging

from indication_scout.constants import CHEMBL_BASE_URL
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.models.model_chembl import MoleculeData

logger = logging.getLogger(__name__)


class ChEMBLClient(BaseClient):
    """Client for querying ChEMBL database."""

    @property
    def _source_name(self) -> str:
        return "chembl"

    async def get_molecule(self, chembl_id: str) -> MoleculeData:
        """Fetch molecule data by ChEMBL ID.

        Hits GET /molecule/{chembl_id}.json and returns a MoleculeData instance.
        Raises DataSourceError if the molecule is not found or the response is malformed.
        """
        url = f"{CHEMBL_BASE_URL}/molecule/{chembl_id}.json"
        try:
            raw = await self._rest_get(url, params={})
        except DataSourceError:
            raise
        except Exception as e:
            raise DataSourceError(self._source_name, f"Unexpected error fetching {chembl_id}: {e}")

        if not isinstance(raw, dict):
            raise DataSourceError(self._source_name, f"Unexpected response shape for '{chembl_id}'")

        return MoleculeData(
            molecule_chembl_id=raw["molecule_chembl_id"],
            molecule_type=raw["molecule_type"],
            max_phase=raw["max_phase"],
            atc_classifications=raw.get("atc_classifications") or [],
            black_box_warning=raw["black_box_warning"],
            first_approval=raw.get("first_approval"),
            oral=raw["oral"],
        )
