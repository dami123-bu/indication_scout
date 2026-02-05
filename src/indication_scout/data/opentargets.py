import httpx

from indication_scout.models.drug import Drug, DrugActivity
from indication_scout.models.indication import DiseaseIndication
from indication_scout.models.target import Target


class OpenTargetsClient:
    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )

    async def _execute(self, query: str, variables: dict) -> dict:
        response = await self.client.post(
            self.BASE_URL,
            json={"query": query, "variables": variables},
        )
        response.raise_for_status()
        return response.json().get("data", {})

    def _parse_target(self, d: dict) -> Target:
        return Target(
            ensembl_id=d["id"],
            symbol=d["approvedSymbol"],
            name=d["approvedName"]
        )

    def _parse_indication(self, d: dict) -> DiseaseIndication:
        return DiseaseIndication(
            efo_id=d["id"],
            name=d["name"],
            therapeutic_areas=[ta["name"] for ta in d.get("therapeuticAreas") or []],
        )

    def _parse_drug_activities(self, d: dict) -> list[DrugActivity]:
        drug_activities = []

        for moa in (d.get("mechanismsOfAction") or {}).get("rows", []):
            targets = [self._parse_target(t) for t in moa.get("targets") or []]
            if targets:
                for target in targets:
                    drug_activities.append(DrugActivity(
                        description=moa.get("mechanismOfAction"),
                        action_type=moa.get("actionType"),
                        target=target,
                    ))
            else:
                drug_activities.append(DrugActivity(
                    description=moa.get("mechanismOfAction"),
                    action_type=moa.get("actionType"),
                ))

        for row in (d.get("indications") or {}).get("rows", []):
            disease = row.get("disease")
            if disease:
                drug_activities.append(DrugActivity(
                    indication=self._parse_indication(disease),
                ))

        return drug_activities

    def _parse_drug(self, d: dict) -> Drug:
        return Drug(
            chembl_id=d["id"],
            generic_name=d["name"],
            description=d.get("description"),
            drug_type=d.get("drugType"),
            is_approved=d.get("isApproved", False),
            has_been_withdrawn=d.get("hasBeenWithdrawn", False),
            year_first_approved=d.get("yearOfFirstApproval"),
            max_clinical_phase=d.get("maximumClinicalTrialPhase"),
            synonyms=d.get("synonyms") or [],
            trade_names=d.get("tradeNames") or [],
            activities=self._parse_drug_activities(d),
        )

    async def get_drug(self, chembl_id: str) -> Drug | None:
        """Get drug information by ChEMBL ID."""
        query = """
        query Drug($chemblId: String!) {
            drug(chemblId: $chemblId) {
                id
                name
                description
                drugType
                hasBeenWithdrawn
                yearOfFirstApproval
                maximumClinicalTrialPhase
                isApproved
                synonyms
                tradeNames
                mechanismsOfAction {
                    rows {
                        mechanismOfAction
                        targetName
                        actionType
                        targets { id approvedSymbol approvedName }
                    }
                }
                indications {
                    rows {
                        disease {
                            id
                            name
                            therapeuticAreas { id name }
                        }
                    }
                }
            }
        }
        """
        data = await self._execute(query, {"chemblId": chembl_id})
        drug_data = data.get("drug")
        if drug_data is None:
            return None

        return self._parse_drug(drug_data)

    async def search(self, term: str, entity_type: str = None) -> list:
        """
        Convert human-readable name to Open Targets ID.

        entity_type: "drug", "disease", "target"
        """
        # Map user-friendly names to Open Targets entity names
        entity_map = {
            "drug": "drug",
            "disease": "disease",
            "target": "target",
        }

        query = """
        query Search($term: String!, $entityNames: [String!]) {
            search(queryString: $term, entityNames: $entityNames, page: {size: 5, index: 0}) {
                hits {
                    id
                    name
                    entity
                    description
                }
            }
        }
        """

        variables = {"term": term}
        if entity_type:
            mapped = entity_map.get(entity_type, entity_type)
            variables["entityNames"] = [mapped]

        data = await self._execute(query, variables)
        return data.get("search", {}).get("hits", [])

    async def resolve_id(self, term: str, entity_type: str) -> str | None:
        """Get the top matching ID, or None if not found."""
        results = await self.search(term, entity_type)
        return results[0]["id"] if results else None

    # Stubbed methods, in case implementation is needed in the future

    async def get_drug_indications(self, chembl_id: str) -> list:
        """What is this drug already approved for?"""
        ...

    async def get_disease_targets(self, efo_id: str, min_score: float = 0.3) -> list:
        """What targets are associated with this disease?"""
        ...

    async def get_target_diseases(self, ensembl_id: str, min_score: float = 0.3) -> list:
        """What diseases are associated with this target?"""
        ...

    async def get_drug_mechanisms(self, chembl_id: str) -> dict:
        """What targets does this drug hit, and how?"""
        ...