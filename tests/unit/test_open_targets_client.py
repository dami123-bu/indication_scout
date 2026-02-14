"""Unit tests for OpenTargetsClient."""

import hashlib
import json
from pathlib import Path

from indication_scout.data_sources.open_targets import OpenTargetsClient


class TestOpenTargetsClientConfig:
    """Tests for OpenTargetsClient configuration."""

    def test_default_config(self):
        """Test that client uses default settings."""
        client = OpenTargetsClient()

        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.cache_dir == Path("_cache")


class TestParseTargetData:
    """Unit tests for _parse_target_data."""

    def setup_method(self) -> None:
        self.client = OpenTargetsClient()

    def test_parses_complete_target(self):
        """Test _parse_target_data correctly reshapes raw API response into TargetData."""
        raw = {
            "id": "ENSG00000141736",
            "approvedSymbol": "ERBB2",
            "approvedName": "erb-b2 receptor tyrosine kinase 2",
            "associatedDiseases": {
                "rows": [
                    {
                        "disease": {
                            "id": "EFO_0000305",
                            "name": "breast carcinoma",
                            "therapeuticAreas": [
                                {"name": "cancer or benign tumor"},
                            ],
                        },
                        "score": 0.70,
                        "datatypeScores": [
                            {"id": "genetic_association", "score": 0.5},
                            {"id": "literature", "score": 0.3},
                        ],
                    }
                ]
            },
            "pathways": [
                {
                    "pathwayId": "R-HSA-9665233",
                    "pathway": "Resistance of ERBB2 KD mutants to trastuzumab",
                    "topLevelTerm": "Disease",
                }
            ],
            "interactions": {
                "rows": [
                    {
                        "targetB": {
                            "id": "ENSG00000124181",
                            "approvedSymbol": "PLCG1",
                        },
                        "score": 0.99,
                        "sourceDatabase": "string",
                        "intBBiologicalRole": "unspecified role",
                        "count": 4,
                    }
                ]
            },
            "knownDrugs": {
                "rows": [
                    {
                        "drugId": "CHEMBL1201585",
                        "prefName": "TRASTUZUMAB",
                        "diseaseId": "EFO_0000305",
                        "label": "breast carcinoma",
                        "phase": 4.0,
                        "status": None,
                        "mechanismOfAction": "Receptor protein-tyrosine kinase erbB-2 inhibitor",
                        "ctIds": ["NCT00001", "NCT00002"],
                    }
                ]
            },
            "expressions": [
                {
                    "tissue": {
                        "id": "UBERON_0002107",
                        "label": "liver",
                        "anatomicalSystems": ["endocrine system"],
                    },
                    "rna": {"value": 11819.0, "level": 5, "unit": "TPM"},
                    "protein": {
                        "level": 2,
                        "reliability": True,
                        "cellType": [
                            {"name": "hepatocytes", "level": 1, "reliability": True}
                        ],
                    },
                }
            ],
            "mousePhenotypes": [
                {
                    "modelPhenotypeId": "MP:0013279",
                    "modelPhenotypeLabel": "increased fasting circulating glucose level",
                    "modelPhenotypeClasses": [
                        {"label": "homeostasis/metabolism phenotype"}
                    ],
                    "biologicalModels": [
                        {
                            "allelicComposition": "Erbb2<tm1> hom",
                            "geneticBackground": "C57BL/6N",
                            "literature": [],
                            "id": "MGI:001",
                        }
                    ],
                }
            ],
            "safetyLiabilities": [
                {
                    "event": "cardiac arrhythmia",
                    "eventId": "EFO_0004269",
                    "datasource": "Lynch et al. (2017)",
                    "literature": "28216264",
                    "url": None,
                    "effects": [
                        {"direction": "Inhibition/Decrease/Downregulation", "dosing": "acute"}
                    ],
                }
            ],
            "geneticConstraint": [
                {
                    "constraintType": "lof",
                    "oe": 0.41,
                    "oeLower": 0.33,
                    "oeUpper": 0.51,
                    "score": 0.06,
                    "upperBin": 1,
                }
            ],
        }

        target = self.client._parse_target_data(raw)

        # Top-level fields
        assert target.target_id == "ENSG00000141736"
        assert target.symbol == "ERBB2"
        assert target.name == "erb-b2 receptor tyrosine kinase 2"

        # Association
        assert len(target.associations) == 1
        assoc = target.associations[0]
        assert assoc.disease_id == "EFO_0000305"
        assert assoc.disease_name == "breast carcinoma"
        assert assoc.overall_score == 0.70
        assert assoc.datatype_scores == {"genetic_association": 0.5, "literature": 0.3}
        assert assoc.therapeutic_areas == ["cancer or benign tumor"]

        # Pathway
        assert len(target.pathways) == 1
        pw = target.pathways[0]
        assert pw.pathway_id == "R-HSA-9665233"
        assert pw.pathway_name == "Resistance of ERBB2 KD mutants to trastuzumab"
        assert pw.top_level_pathway == "Disease"

        # Interaction
        assert len(target.interactions) == 1
        inter = target.interactions[0]
        assert inter.interacting_target_id == "ENSG00000124181"
        assert inter.interacting_target_symbol == "PLCG1"
        assert inter.interaction_score == 0.99
        assert inter.source_database == "string"
        assert inter.biological_role == "unspecified role"
        assert inter.evidence_count == 4
        assert inter.interaction_type == "functional"

        # DrugSummary
        assert len(target.drug_summaries) == 1
        ds = target.drug_summaries[0]
        assert ds.drug_id == "CHEMBL1201585"
        assert ds.drug_name == "TRASTUZUMAB"
        assert ds.disease_id == "EFO_0000305"
        assert ds.disease_name == "breast carcinoma"
        assert ds.phase == 4.0
        assert ds.status is None
        assert ds.mechanism_of_action == "Receptor protein-tyrosine kinase erbB-2 inhibitor"
        assert ds.clinical_trial_ids == ["NCT00001", "NCT00002"]

        # Expression
        assert len(target.expressions) == 1
        expr = target.expressions[0]
        assert expr.tissue_id == "UBERON_0002107"
        assert expr.tissue_name == "liver"
        assert expr.tissue_anatomical_system == "endocrine system"
        assert expr.rna.value == 11819.0
        assert expr.rna.quantile == 5
        assert expr.rna.unit == "TPM"
        assert expr.protein.level == 2
        assert expr.protein.reliability is True
        assert len(expr.protein.cell_types) == 1
        assert expr.protein.cell_types[0].name == "hepatocytes"
        assert expr.protein.cell_types[0].level == 1
        assert expr.protein.cell_types[0].reliability is True

        # MousePhenotype
        assert len(target.mouse_phenotypes) == 1
        pheno = target.mouse_phenotypes[0]
        assert pheno.phenotype_id == "MP:0013279"
        assert pheno.phenotype_label == "increased fasting circulating glucose level"
        assert pheno.phenotype_categories == ["homeostasis/metabolism phenotype"]
        assert len(pheno.biological_models) == 1
        assert pheno.biological_models[0].allelic_composition == "Erbb2<tm1> hom"
        assert pheno.biological_models[0].genetic_background == "C57BL/6N"

        # SafetyLiability
        assert len(target.safety_liabilities) == 1
        sl = target.safety_liabilities[0]
        assert sl.event == "cardiac arrhythmia"
        assert sl.event_id == "EFO_0004269"
        assert sl.datasource == "Lynch et al. (2017)"
        assert sl.literature == "28216264"
        assert sl.url is None
        assert len(sl.effects) == 1
        assert sl.effects[0].direction == "Inhibition/Decrease/Downregulation"
        assert sl.effects[0].dosing == "acute"

        # GeneticConstraint
        assert len(target.genetic_constraint) == 1
        gc = target.genetic_constraint[0]
        assert gc.constraint_type == "lof"
        assert gc.oe == 0.41
        assert gc.oe_lower == 0.33
        assert gc.oe_upper == 0.51
        assert gc.score == 0.06
        assert gc.upper_bin == 1


class TestCacheKeyGeneration:
    """Tests for _cache_key determinism and uniqueness."""

    def setup_method(self) -> None:
        self.client = OpenTargetsClient()

    def test_deterministic_same_inputs(self) -> None:
        """Same namespace + params always produce the same key."""
        key1 = self.client._cache_key("drug", {"chembl_id": "CHEMBL25"})
        key2 = self.client._cache_key("drug", {"chembl_id": "CHEMBL25"})

        assert key1 == key2

    def test_different_namespace_different_key(self) -> None:
        """Different namespaces with identical params produce different keys."""
        key_drug = self.client._cache_key("drug", {"id": "CHEMBL25"})
        key_target = self.client._cache_key("target", {"id": "CHEMBL25"})

        assert key_drug != key_target

    def test_different_params_different_key(self) -> None:
        """Same namespace with different params produce different keys."""
        key1 = self.client._cache_key("drug", {"chembl_id": "CHEMBL25"})
        key2 = self.client._cache_key("drug", {"chembl_id": "CHEMBL1431"})

        assert key1 != key2

    def test_param_order_does_not_affect_key(self) -> None:
        """Dict key ordering should not change the hash (sort_keys=True)."""
        key1 = self.client._cache_key("drug", {"a": "1", "b": "2"})
        key2 = self.client._cache_key("drug", {"b": "2", "a": "1"})

        assert key1 == key2

    def test_key_is_valid_sha256_hex(self) -> None:
        """Key should be a 64-character lowercase hex string (SHA-256)."""
        key = self.client._cache_key("drug", {"chembl_id": "CHEMBL25"})

        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_matches_expected_sha256(self) -> None:
        """Key matches a manually computed SHA-256 of the canonical JSON."""
        namespace = "drug"
        params = {"chembl_id": "CHEMBL25"}
        raw = json.dumps({"ns": namespace, **params}, sort_keys=True, default=str)
        expected = hashlib.sha256(raw.encode()).hexdigest()

        key = self.client._cache_key(namespace, params)

        assert key == expected
