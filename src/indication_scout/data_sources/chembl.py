"""ChEMBL API client."""

import logging

from pathlib import Path

from indication_scout.constants import CACHE_TTL, CHEMBL_BASE_URL, DEFAULT_CACHE_DIR
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
from indication_scout.models.model_chembl import ATCDescription, MoleculeData, MoleculeSynonym
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)


class ChEMBLClient(BaseClient):
    """Client for querying ChEMBL database."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        super().__init__()
        self.cache_dir = cache_dir

    @property
    def _source_name(self) -> str:
        return "chembl"

    async def get_atc_description(self, atc_code: str) -> ATCDescription:
        """Fetch ATC classification hierarchy for a single ATC code.

        Hits GET /atc_class/{atc_code}.json and returns an ATCDescription instance.
        Results are cached under namespace "atc_description" for CACHE_TTL seconds.
        Raises DataSourceError if the code is not found or the response is malformed.
        """
        cached = cache_get("atc_description", {"atc_code": atc_code}, self.cache_dir)
        if cached is not None:
            return ATCDescription.model_validate(cached)

        url = f"{CHEMBL_BASE_URL}/atc_class/{atc_code}.json"
        try:
            raw = await self._rest_get(url, params={})
        except Exception as e:
            raise DataSourceError(
                self._source_name,
                f"Unexpected error fetching ATC code '{atc_code}': {e}",
            )

        if not isinstance(raw, dict):
            raise DataSourceError(
                self._source_name,
                f"Unexpected response shape for ATC code '{atc_code}'",
            )

        result = ATCDescription(
            level1=raw["level1"],
            level1_description=raw["level1_description"],
            level2=raw["level2"],
            level2_description=raw["level2_description"],
            level3=raw["level3"],
            level3_description=raw["level3_description"],
            level4=raw["level4"],
            level4_description=raw["level4_description"],
            level5=raw["level5"],
            who_name=raw["who_name"],
        )

        cache_set(
            "atc_description",
            {"atc_code": atc_code},
            result.model_dump(),
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        return result

    async def get_molecule(self, chembl_id: str) -> MoleculeData:
        """Fetch molecule data by ChEMBL ID.

        Hits GET /molecule/{chembl_id}.json and returns a MoleculeData instance.
        Raises DataSourceError if the molecule is not found or the response is malformed.
        """
        url = f"{CHEMBL_BASE_URL}/molecule/{chembl_id}.json"
        try:
            raw = await self._rest_get(url, params={})
        except Exception as e:
            raise DataSourceError(
                self._source_name, f"Unexpected error fetching {chembl_id}: {e}"
            )

        if not isinstance(raw, dict):
            raise DataSourceError(
                self._source_name, f"Unexpected response shape for '{chembl_id}'"
            )

        synonyms = [
            MoleculeSynonym(
                molecule_synonym=s.get("molecule_synonym", ""),
                syn_type=s.get("syn_type", ""),
                synonyms=s.get("synonyms", ""),
            )
            for s in raw.get("molecule_synonyms") or []
        ]

        return MoleculeData(
            molecule_chembl_id=raw["molecule_chembl_id"],
            molecule_type=raw["molecule_type"],
            max_phase=raw["max_phase"],
            atc_classifications=raw.get("atc_classifications") or [],
            black_box_warning=raw["black_box_warning"],
            first_approval=raw.get("first_approval"),
            oral=raw["oral"],
            molecule_synonyms=synonyms,
        )

    async def get_trade_names(self, chembl_id: str) -> list[str]:
        """Fetch trade names for a drug from ChEMBL, including salt forms.

        For small molecules, trade names often live on salt forms (e.g. HCl,
        HBr) rather than the parent molecule. This method fetches the parent
        molecule's synonyms, discovers salt forms via the molecule_hierarchy
        endpoint, and collects TRADE_NAME synonyms from all of them.

        Filters out "component of" entries (combination products).
        Results are cached under namespace "chembl_trade_names".

        Args:
            chembl_id: ChEMBL ID of the parent molecule (e.g. "CHEMBL894").

        Returns:
            Deduplicated list of trade names.
        """
        cached = cache_get(
            "chembl_trade_names", {"chembl_id": chembl_id}, self.cache_dir
        )
        if cached is not None:
            return cached

        trade_names: list[str] = []

        # 1. Get trade names from the parent molecule itself
        parent = await self.get_molecule(chembl_id)
        trade_names.extend(
            s.molecule_synonym
            for s in parent.molecule_synonyms
            if s.syn_type == "TRADE_NAME"
        )

        # 2. Discover salt forms via molecule_hierarchy
        hierarchy_url = f"{CHEMBL_BASE_URL}/molecule.json"
        try:
            hierarchy_raw = await self._rest_get(
                hierarchy_url,
                params={
                    "molecule_hierarchy__parent_chembl_id": chembl_id,
                    "limit": 50,
                },
            )
        except DataSourceError as e:
            logger.warning(
                "Failed to fetch salt forms for %s: %s", chembl_id, e
            )
            hierarchy_raw = {}

        salt_molecules = (hierarchy_raw or {}).get("molecules", [])
        for mol in salt_molecules:
            mol_id = mol.get("molecule_chembl_id", "")
            if mol_id == chembl_id:
                continue  # skip the parent, already processed
            for s in mol.get("molecule_synonyms") or []:
                if s.get("syn_type") == "TRADE_NAME":
                    trade_names.append(s.get("molecule_synonym", ""))

        # 3. Filter out "component of" entries and deduplicate
        filtered = [
            name
            for name in trade_names
            if name and "component of" not in name.lower()
        ]
        result = list(dict.fromkeys(filtered))

        logger.info(
            "ChEMBL trade names for %s: %d found", chembl_id, len(result)
        )

        cache_set(
            "chembl_trade_names",
            {"chembl_id": chembl_id},
            result,
            self.cache_dir,
            ttl=CACHE_TTL,
        )
        return result
