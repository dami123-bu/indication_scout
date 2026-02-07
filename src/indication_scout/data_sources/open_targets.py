from indication_scout.data_sources.base_client import (
    DataSourceError,
    RequestContext,
    ClientConfig,
    BaseClient,
)


from indication_scout.models.open_targets import (
    Association,
    Pathway,
    Interaction,
    DrugSummary,
    TissueExpression,
    MousePhenotype,
    TargetData,
    GeneticConstraint,
    AdverseEvent,
    CellTypeExpression,
    RNAExpression,
    BiologicalModel,
    DrugData,
    DrugTarget,
    ProteinExpression,
    DrugWarning,
    Indication,
    SafetyLiability,
    SafetyEffect,
    DiseaseDrug,
)

INTERACTION_TYPE_MAP = {
    "intact": "physical",
    "signor": "signalling",
    "reactome": "enzymatic",
    "string": "functional",
}


class OpenTargetsClient(BaseClient):
    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    CACHE_TTL = 5 * 86400

    def __init__(self, config: ClientConfig | None = None):
        super().__init__(config)
        self._drug_cache: dict[str, DrugData] = {}  # keyed by chembl_id
        self._target_data_cache: dict[str, TargetData] = {}  # keyed by target_id

    @property
    def _source_name(self) -> str:
        return "open_targets"

    # ------------------------------------------------------------------
    # Public: get_drug and get_target
    # ------------------------------------------------------------------

    async def get_drug(self, drug_name: str) -> DrugData:
        """Fetch drug data by name. Returns cached data if available."""
        chembl_id = await self._resolve_drug_name(drug_name)

        if chembl_id in self._drug_cache:
            return self._drug_cache[chembl_id]

        disk_hit = await self.cache.get("drug", {"chembl_id": chembl_id})
        if disk_hit:
            drug_data = DrugData.model_validate(disk_hit)
            self._drug_cache[chembl_id] = drug_data
            return drug_data

        drug_data = await self._fetch_drug(chembl_id)

        self._drug_cache[chembl_id] = drug_data
        await self.cache.set(
            "drug",
            {"chembl_id": chembl_id},
            drug_data.model_dump(),
            ttl=self.CACHE_TTL,
        )

        return drug_data

    async def get_drug_indications(self, drug_name: str) -> list[Indication]:
        drug = await self.get_drug(drug_name)
        return drug.indications

    async def get_target_data(self, target_id: str) -> TargetData:
        """Fetch target data by ID. Returns cached data if available."""
        if target_id in self._target_data_cache:
            return self._target_data_cache[target_id]

        disk_hit = await self.cache.get("target", {"target_id": target_id})
        if disk_hit:
            target_data = TargetData.model_validate(disk_hit)
            self._target_data_cache[target_id] = target_data
            return target_data

        target_data = await self._fetch_target(target_id)

        self._target_data_cache[target_id] = target_data
        await self.cache.set(
            "target",
            {"target_id": target_id},
            target_data.model_dump(),
            ttl=self.CACHE_TTL,
        )

        return target_data

    # ------------------------------------------------------------------
    # Public accessors — convenience methods using get_drug/get_target
    # ------------------------------------------------------------------

    async def get_target_data_associations(
        self, target_id: str, min_score: float = 0.1
    ) -> list[Association]:
        target = await self.get_target_data(target_id)
        return [a for a in target.associations if a.overall_score >= min_score]

    async def get_target_data_pathways(self, target_id: str) -> list[Pathway]:
        target = await self.get_target_data(target_id)
        return target.pathways

    async def get_target_data_interactions(self, target_id: str) -> list[Interaction]:
        target = await self.get_target_data(target_id)
        return target.interactions

    async def get_target_data_drug_summaries(self, target_id: str) -> list[DrugSummary]:
        target = await self.get_target_data(target_id)
        return target.drug_summaries

    async def get_target_data_tissue_expression(
        self, target_id: str
    ) -> list[TissueExpression]:
        target = await self.get_target_data(target_id)
        return target.expressions

    async def get_target_data_mouse_phenotypes(
        self, target_id: str
    ) -> list[MousePhenotype]:
        target = await self.get_target_data(target_id)
        return target.mouse_phenotypes

    async def get_target_data_safety_liabilities(
        self, target_id: str
    ) -> list[SafetyLiability]:
        target = await self.get_target_data(target_id)
        return target.safety_liabilities

    async def get_target_data_genetic_constraints(
        self, target_id: str
    ) -> list[GeneticConstraint]:
        target = await self.get_target_data(target_id)
        return target.genetic_constraint

    # ------------------------------------------------------------------
    # Private: network calls
    # ------------------------------------------------------------------

    async def _resolve_drug_name(self, name: str) -> str:
        """Search by name → return ChEMBL ID."""
        result = await self._graphql(
            self.BASE_URL,
            SEARCH_QUERY,
            variables={"q": name},
            cache_namespace="drug_name_resolve",
            cache_ttl=self.CACHE_TTL,
            context=self._ctx("resolve_drug_name"),
        )
        if not result.is_complete:
            raise DataSourceError(
                self._source_name,
                f"Failed to resolve drug name '{name}': {result.errors}",
            )
        hits = result.data["data"]["search"]["hits"]
        drug_hits = [h for h in hits if h["entity"] == "drug"]
        if not drug_hits:
            raise DataSourceError(
                self._source_name,
                f"No drug found for '{name}'",
            )
        return drug_hits[0]["id"]

    async def _fetch_drug(self, chembl_id: str) -> DrugData:
        """Fetch full drug node from Open Targets."""
        result = await self._graphql(
            self.BASE_URL,
            DRUG_QUERY,
            variables={"id": chembl_id},
            context=self._ctx("fetch_drug"),
        )
        if not result.is_complete:
            raise DataSourceError(
                self._source_name,
                f"Failed to fetch drug {chembl_id}: {result.errors}",
            )
        return self._parse_drug_data(result.data["data"]["drug"])

    async def _fetch_target(self, target_id: str) -> TargetData:
        """Fetch full target node. Paginates associations if needed."""
        result = await self._graphql(
            self.BASE_URL,
            TARGET_QUERY,
            variables={"id": target_id},
            context=self._ctx("fetch_target"),
        )
        if not result.is_complete:
            raise DataSourceError(
                self._source_name,
                f"Failed to fetch target {target_id}: {result.errors}",
            )
        target_data = self._parse_target_data(result.data["data"]["target"])

        # Paginate if we hit the association page limit
        if len(target_data.associations) >= 500:
            target_data.associations = await self._paginate_associations(target_id)

        return target_data

    async def _paginate_associations(self, target_id: str) -> list[Association]:
        """Fetch all associations when count exceeds single page."""
        all_associations = []
        page_index = 0
        page_size = 500

        while True:
            result = await self._graphql(
                self.BASE_URL,
                ASSOCIATIONS_PAGE_QUERY,
                variables={
                    "id": target_id,
                    "index": page_index,
                    "size": page_size,
                },
                context=self._ctx("paginate_associations"),
            )
            if not result.is_complete:
                break

            rows = result.data["data"]["target"]["associatedDiseases"]["rows"]
            all_associations.extend(self._parse_association(r) for r in rows)

            if len(rows) < page_size:
                break
            page_index += 1

        return all_associations

    # ------------------------------------------------------------------
    # Parsers: raw GraphQL response → Pydantic models
    # ------------------------------------------------------------------

    def _parse_drug_data(self, raw: dict) -> DrugData:
        targets = []
        for row in raw.get("mechanismsOfAction", {}).get("rows", []):
            for t in row.get("targets", []):
                targets.append(
                    DrugTarget(
                        target_id=t["id"],
                        target_symbol=t["approvedSymbol"],
                        mechanism_of_action=row["mechanismOfAction"],
                        action_type=row.get("actionType"),
                    )
                )

        warnings = [
            DrugWarning(
                warning_type=w.get("warningType", ""),
                description=w.get("description"),
                toxicity_class=w.get("toxicityClass"),
                country=w.get("country"),
                year=w.get("year"),
                efo_id=w.get("efoId"),
            )
            for w in raw.get("drugWarnings", [])
        ]

        indications = [
            Indication(
                disease_id=row["disease"]["id"],
                disease_name=row["disease"]["name"],
                max_phase=row["maxPhaseForIndication"],
                references=row.get("references", []),
            )
            for row in raw.get("indications", {}).get("rows", [])
        ]

        adverse_events = [
            self._parse_adverse_event(ae)
            for ae in raw.get("adverseEvents", {}).get("rows", [])
        ]

        return DrugData(
            chembl_id=raw["id"],
            name=raw["name"],
            synonyms=raw.get("synonyms", []),
            trade_names=raw.get("tradeNames", []),
            drug_type=raw.get("drugType", ""),
            is_approved=raw.get("isApproved", False),
            max_clinical_phase=raw.get("maximumClinicalTrialPhase", 0),
            year_first_approved=raw.get("yearOfFirstApproval"),
            warnings=warnings,
            indications=indications,
            targets=targets,
            adverse_events=adverse_events,
            adverse_events_critical_value=raw.get("adverseEvents", {}).get(
                "criticalValue", 0.0
            ),
        )

    def _parse_target_data(self, raw: dict) -> TargetData:
        return TargetData(
            target_id=raw["id"],
            symbol=raw["approvedSymbol"],
            name=raw.get("approvedName", ""),
            associations=[
                self._parse_association(r)
                for r in raw.get("associatedDiseases", {}).get("rows", [])
            ],
            pathways=[self._parse_pathway(p) for p in raw.get("pathways", [])],
            interactions=[
                self._parse_interaction(i)
                for i in raw.get("interactions", {}).get("rows", [])
            ],
            drug_summaries=[
                self._parse_drug_summary(d)
                for d in raw.get("knownDrugs", {}).get("rows", [])
            ],
            expressions=[self._parse_expression(e) for e in raw.get("expressions", [])],
            mouse_phenotypes=[
                self._parse_phenotype(p) for p in raw.get("mousePhenotypes", [])
            ],
            safety_liabilities=[
                self._parse_safety_liability(sl)
                for sl in raw.get("safetyLiabilities", [])
            ],
            genetic_constraint=[
                self._parse_constraint(c) for c in raw.get("geneticConstraint", [])
            ],
        )

    def _parse_association(self, raw: dict) -> Association:
        datatype_scores = {s["id"]: s["score"] for s in raw.get("datatypeScores", [])}
        disease = raw["disease"]
        therapeutic_areas = [ta["name"] for ta in disease.get("therapeuticAreas", [])]
        return Association(
            disease_id=disease["id"],
            disease_name=disease["name"],
            overall_score=raw["score"],
            datatype_scores=datatype_scores,
            therapeutic_areas=therapeutic_areas,
        )

    def _parse_pathway(self, raw: dict) -> Pathway:
        return Pathway(
            pathway_id=raw.get("pathwayId", ""),
            pathway_name=raw.get("pathway", ""),
            top_level_pathway=raw.get("topLevelTerm", ""),
        )

    def _parse_interaction(self, raw: dict) -> Interaction:
        source = raw.get("sourceDatabase", "")
        target_b = raw.get("targetB", {}) or {}
        return Interaction(
            interacting_target_id=target_b.get("id", raw.get("intB", "")),
            interacting_target_symbol=target_b.get("approvedSymbol", ""),
            interaction_score=raw.get("score", 0.0),
            source_database=source,
            biological_role=raw.get("intBBiologicalRole", ""),
            evidence_count=raw.get("count", 0),
            interaction_type=INTERACTION_TYPE_MAP.get(source.lower()),
        )

    def _parse_drug_summary(self, raw: dict) -> DrugSummary:
        return DrugSummary(
            drug_id=raw.get("drugId", ""),
            drug_name=raw.get("prefName", ""),
            disease_id=raw.get("diseaseId", ""),
            disease_name=raw.get("label", ""),
            phase=raw.get("phase", 0),
            status=raw.get("status"),
            mechanism_of_action=raw.get("mechanismOfAction", ""),
            clinical_trial_ids=raw.get("ctIds", []),
        )

    def _parse_expression(self, raw: dict) -> TissueExpression:
        tissue = raw.get("tissue", {})
        rna = raw.get("rna", {})
        protein = raw.get("protein", {})
        anatomical_systems = tissue.get("anatomicalSystems", [])
        cell_types = [
            CellTypeExpression(
                name=ct["name"],
                level=ct["level"],
                reliability=ct.get("reliability", False),
            )
            for ct in protein.get("cellType", []) or []
        ]
        return TissueExpression(
            tissue_id=tissue.get("id", ""),
            tissue_name=tissue.get("label", ""),
            tissue_anatomical_system=(
                anatomical_systems[0] if anatomical_systems else ""
            ),
            rna=RNAExpression(
                value=rna.get("value", 0.0),
                quantile=rna.get("level", 0),
                unit=rna.get("unit", "TPM"),
            ),
            protein=ProteinExpression(
                level=protein.get("level", 0),
                reliability=protein.get("reliability", False),
                cell_types=cell_types,
            ),
        )

    def _parse_phenotype(self, raw: dict) -> MousePhenotype:
        categories = [c["label"] for c in raw.get("modelPhenotypeClasses", [])]
        models = [
            BiologicalModel(
                allelic_composition=m.get("allelicComposition") or "",
                genetic_background=m.get("geneticBackground") or "",
                literature=m.get("literature") or [],
                model_id=m.get("id") or "",
            )
            for m in raw.get("biologicalModels") or []
        ]
        return MousePhenotype(
            phenotype_id=raw.get("modelPhenotypeId") or "",
            phenotype_label=raw.get("modelPhenotypeLabel") or "",
            phenotype_categories=categories,
            biological_models=models,
        )

    def _parse_adverse_event(self, raw: dict) -> AdverseEvent:
        return AdverseEvent(
            name=raw["name"],
            meddra_code=raw.get("meddraCode"),
            count=raw["count"],
            log_likelihood_ratio=raw["logLR"],
        )

    def _parse_safety_liability(self, raw: dict) -> SafetyLiability:
        effects = [
            SafetyEffect(
                direction=e.get("direction", ""),
                dosing=e.get("dosing"),
            )
            for e in raw.get("effects", [])
        ]
        return SafetyLiability(
            event=raw.get("event"),
            event_id=raw.get("eventId"),
            effects=effects,
            datasource=raw.get("datasource"),
            literature=raw.get("literature"),
            url=raw.get("url"),
        )

    def _parse_constraint(self, raw: dict) -> GeneticConstraint:
        return GeneticConstraint(
            constraint_type=raw["constraintType"],
            oe=raw.get("oe"),
            oe_lower=raw.get("oeLower"),
            oe_upper=raw.get("oeUpper"),
            score=raw.get("score"),
            upper_bin=raw.get("upperBin"),
        )

    async def get_disease_drugs(self, disease_id: str) -> list[DrugSummary]:
        """All drugs for a disease, any target, any mechanism."""
        result = await self._graphql(
            self.BASE_URL,
            DISEASE_DRUGS_QUERY,
            variables={"id": disease_id, "size": 200},
            cache_namespace=f"disease_drugs:{disease_id}",
            cache_ttl=self.CACHE_TTL,
            context=self._ctx("get_disease_drugs"),
        )
        if not result.is_complete:
            raise DataSourceError(
                self._source_name,
                f"Failed to fetch drugs for disease {disease_id}: {result.errors}",
            )
        return self._parse_disease_drugs(result.data["data"])

    def _parse_disease_drugs(self, data: dict) -> list[DrugSummary]:
        """Parse and deduplicate disease drugs — one entry per drug, highest phase wins."""
        disease = data.get("disease") or {}
        rows = disease.get("knownDrugs", {}).get("rows", [])

        by_drug: dict[str, DrugSummary] = {}
        for row in rows:
            drug = self._parse_drug_summary(row)
            if drug.drug_id not in by_drug or drug.phase > by_drug[drug.drug_id].phase:
                by_drug[drug.drug_id] = drug
        return list(by_drug.values())

    def _ctx(self, method: str) -> RequestContext:
        return RequestContext(source=self._source_name, method=method)


# ------------------------------------------------------------------
# GraphQL queries
# ------------------------------------------------------------------

SEARCH_QUERY = """
query($q: String!) {
    search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 1}) {
        hits { id entity }
    }
}
"""

DRUG_QUERY = """
query($id: String!) {
    drug(chemblId: $id) {
        id name synonyms tradeNames drugType
        isApproved maximumClinicalTrialPhase yearOfFirstApproval

        mechanismsOfAction {
            rows {
                mechanismOfAction actionType
                targets { id approvedSymbol }
            }
        }

        indications {
            approvedIndications
            rows {
                maxPhaseForIndication
                disease { id name }
                references { source ids }
            }
        }

        drugWarnings {
            warningType description toxicityClass
            country year efoId efoTerm
        }

        adverseEvents(page: {index: 0, size: 100}) {
            rows { name meddraCode count logLR }
            criticalValue
        }
    }
}
"""

TARGET_QUERY = """
query($id: String!) {
    target(ensemblId: $id) {
        id approvedSymbol approvedName

        associatedDiseases(page: {index: 0, size: 500}) {
            rows {
                disease {
                    id name
                    therapeuticAreas { id name }
                }
                score
                datatypeScores { id score }
            }
        }

        pathways { pathwayId pathway topLevelTerm }

        interactions(page: {index: 0, size: 200}) {
            rows {
                intB intBBiologicalRole score
                sourceDatabase count
                targetB { id approvedSymbol }
            }
        }

        knownDrugs(size: 200) {
            rows {
                drugId prefName diseaseId label
                phase status mechanismOfAction ctIds
            }
        }

        expressions {
            tissue {
                id label
                anatomicalSystems
            }
            rna { value unit level }
            protein {
                level reliability
                cellType { name level reliability }
            }
        }

        mousePhenotypes {
            modelPhenotypeId modelPhenotypeLabel
            modelPhenotypeClasses { id label }
            biologicalModels {
                allelicComposition geneticBackground
                id literature
            }
        }

        safetyLiabilities {
            event
            eventId
            effects { direction dosing }
            datasource
            literature
            url
        }

        geneticConstraint {
            constraintType score exp obs
            oe oeLower oeUpper upperBin upperBin6
        }
    }
}
"""

ASSOCIATIONS_PAGE_QUERY = """
query($id: String!, $index: Int!, $size: Int!) {
    target(ensemblId: $id) {
        associatedDiseases(page: {index: $index, size: $size}) {
            rows {
                disease {
                    id name
                    therapeuticAreas { id name }
                }
                score
                datatypeScores { id score }
            }
        }
    }
}
"""

DISEASE_DRUGS_QUERY = """
query($id: String!, $size: Int!) {
    disease(efoId: $id) {
        knownDrugs(size: $size) {
            rows {
                drugId prefName targetId approvedSymbol
                diseaseId label phase status
                mechanismOfAction ctIds
            }
        }
    }
}
"""
