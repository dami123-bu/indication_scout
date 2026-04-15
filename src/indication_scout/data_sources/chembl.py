"""ChEMBL API client."""

import logging

from pathlib import Path

from indication_scout.constants import CACHE_TTL, CHEMBL_BASE_URL, DEFAULT_CACHE_DIR, OPEN_TARGETS_BASE_URL
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
                molecule_synonym=s.get("molecule_synonym", "").lower(),
                syn_type=s.get("syn_type", ""),
                synonyms=s.get("synonyms", "").lower(),
            )
            for s in raw.get("molecule_synonyms") or []
        ]

        hierarchy = raw.get("molecule_hierarchy") or {}

        return MoleculeData(
            molecule_chembl_id=raw["molecule_chembl_id"],
            pref_name=(raw.get("pref_name") or "").lower(),
            parent_chembl_id=hierarchy.get("parent_chembl_id", ""),
            molecule_type=raw["molecule_type"],
            max_phase=raw["max_phase"],
            atc_classifications=raw.get("atc_classifications") or [],
            black_box_warning=raw["black_box_warning"],
            first_approval=raw.get("first_approval"),
            oral=raw["oral"],
            molecule_synonyms=synonyms,
        )

    async def get_all_drug_names(self, chembl_id: str) -> list[str]:
        """Fetch all known names for a drug from ChEMBL.

        Returns a list where the first element is always pref_name (the
        generic/preferred name), followed by trade names from the parent
        molecule and its salt forms. All names are lowercased.

        For small molecules, trade names often live on salt forms (e.g. HCl,
        HBr) rather than the parent molecule. This method fetches the parent
        molecule's synonyms, discovers salt forms via the molecule_hierarchy
        endpoint, and collects TRADE_NAME synonyms from all of them.

        Filters out "component of" entries (combination products).
        Results are cached under namespace "chembl_drug_names".

        Args:
            chembl_id: ChEMBL ID of the parent molecule (e.g. "CHEMBL894").

        Returns:
            Deduplicated list of drug names, pref_name first. All lowercase.
        """
        cached = cache_get(
            "chembl_drug_names", {"chembl_id": chembl_id}, self.cache_dir
        )
        if cached is not None:
            return cached

        trade_names: list[str] = []

        # 1. Get pref_name and trade names from the parent molecule itself
        parent = await self.get_molecule(chembl_id)
        pref_name = parent.pref_name if parent.pref_name else chembl_id

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
                    trade_names.append(s.get("molecule_synonym", "").lower())

        # 3. Filter out "component of" entries and deduplicate with pref_name first
        filtered = [
            name
            for name in trade_names
            if name and "component of" not in name
        ]
        result = [pref_name] + [n for n in list(dict.fromkeys(filtered)) if n != pref_name]

        logger.info(
            "ChEMBL drug names for %s: %d found", chembl_id, len(result)
        )

        cache_set(
            "chembl_drug_names",
            {"chembl_id": chembl_id},
            result,
            self.cache_dir,
            ttl=CACHE_TTL,
        )

        # Reverse index: name → ChEMBL ID so any code can go name → all drug names
        for name in result:
            cache_set(
                "drug_name_to_chembl",
                {"name": name},
                chembl_id,
                self.cache_dir,
                ttl=CACHE_TTL,
            )

        return result


class _OTSearchClient(BaseClient):
    """Minimal client for Open Targets GraphQL search.

    Avoids importing OpenTargetsClient (which imports chembl → circular).
    """

    @property
    def _source_name(self) -> str:
        return "open_targets"


_OT_DRUG_SEARCH_QUERY = """
query($q: String!) {
    search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 1}) {
        hits { id entity }
    }
}
"""


async def resolve_drug_name(drug_name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> str:
    """Resolve a free-text drug name to a canonical (parent) ChEMBL ID.

    1. Searches Open Targets GraphQL for the drug name → some ChEMBL ID.
    2. Fetches the molecule from ChEMBL to read parent_chembl_id.
    3. If the result is a salt (parent differs), follows to the parent.

    Results are cached under namespace "resolve_drug_name".
    Raises DataSourceError if the drug is not found.

    Args:
        drug_name: Free-text drug name (e.g. "metformin", "Glucophage").
        cache_dir: Directory for file-based cache.

    Returns:
        Canonical parent ChEMBL ID (e.g. "CHEMBL1431").
    """
    cached = cache_get("resolve_drug_name", {"drug_name": drug_name.lower()}, cache_dir)
    if cached is not None:
        return cached

    # Step 1: OT search → some ChEMBL ID
    async with _OTSearchClient() as client:
        data = await client._graphql(
            OPEN_TARGETS_BASE_URL, _OT_DRUG_SEARCH_QUERY, {"q": drug_name}
        )

    hits = data["data"]["search"]["hits"]
    drug_hits = [h for h in hits if h["entity"] == "drug"]
    if not drug_hits:
        raise DataSourceError(
            "chembl",
            f"No drug found for '{drug_name}'",
        )
    search_chembl_id = drug_hits[0]["id"]

    # Step 2: Check if this is a salt — follow to parent if so
    async with ChEMBLClient(cache_dir=cache_dir) as client:
        molecule = await client.get_molecule(search_chembl_id)

    parent_id = molecule.parent_chembl_id
    if parent_id and parent_id != search_chembl_id:
        logger.info(
            "Salt detected: %s -> parent %s", search_chembl_id, parent_id
        )
        canonical_id = parent_id
    else:
        canonical_id = search_chembl_id

    cache_set(
        "resolve_drug_name",
        {"drug_name": drug_name.lower()},
        canonical_id,
        cache_dir,
        ttl=CACHE_TTL,
    )

    return canonical_id


async def get_all_drug_names(chembl_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> list[str]:
    """Fetch all known names for a drug, cache-aware.

    Checks the "chembl_drug_names" cache first. On miss, instantiates a
    ChEMBLClient and delegates to ChEMBLClient.get_all_drug_names.

    Args:
        chembl_id: ChEMBL ID of the parent molecule.
        cache_dir: Directory for file-based cache.

    Returns:
        Deduplicated list of drug names, pref_name first. All lowercase.
    """
    cached = cache_get("chembl_drug_names", {"chembl_id": chembl_id}, cache_dir)
    if cached is not None:
        return cached
    async with ChEMBLClient(cache_dir=cache_dir) as client:
        return await client.get_all_drug_names(chembl_id)
