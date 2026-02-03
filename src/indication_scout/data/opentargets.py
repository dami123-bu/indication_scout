import httpx

from indication_scout.models.drug import Drug, Mechanism, DiseaseIndication


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

    async def get_drug_mechanisms(self, chembl_id: str) -> dict:
        """What targets does this drug hit, and how?"""
        ...

    async def get_drug(self, chembl_id: str) -> Drug | None:
        """Get comprehensive drug information by ChEMBL ID."""
        query = """
        query Drug($chemblId: String!) {
            drug(chemblId: $chemblId) {
                id
                name
                description
                drugType
                maximumClinicalTrialPhase
                mechanismsOfAction {
                    rows {
                        mechanismOfAction
                        targetName
                    }
                }
                indications {
                    rows {
                        disease {
                            id
                            name
                        }
                    }
                }
            }
        }
        """
        data = await self._execute(query, {"chemblId": chembl_id})
        drug_data = data.get("drug")
        if not drug_data:
            return None

        mechanisms = [
            Mechanism(
                description=m["mechanismOfAction"],
                target_name=m.get("targetName"),
            )
            for m in drug_data.get("mechanismsOfAction", {}).get("rows", [])
        ]

        indications = [
            DiseaseIndication(
                disease_id=row["disease"]["id"],
                disease_name=row["disease"]["name"],
            )
            for row in drug_data.get("indications", {}).get("rows", [])
            if row.get("disease")
        ]

        return Drug(
            chembl_id=drug_data["id"],
            name=drug_data["name"],
            description=drug_data.get("description"),
            drug_type=drug_data.get("drugType"),
            max_clinical_phase=drug_data.get("maximumClinicalTrialPhase"),
            mechanisms=mechanisms,
            indications=indications,
        )

    async def get_drug_indications(self, chembl_id: str) -> list:
        """What is this drug already approved for?"""
        ...

    async def get_disease_targets(self, efo_id: str, min_score: float = 0.3) -> list:
        """What targets are associated with this disease?"""
        ...

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
