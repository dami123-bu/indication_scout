"""Integration tests for OpenTargetsClient accessor methods."""

import unittest

from indication_scout.data_sources.open_targets import OpenTargetsClient


class TestOpenTargetsAccessors(unittest.IsolatedAsyncioTestCase):
    """Integration tests for OpenTargetsClient accessor methods."""

    async def asyncSetUp(self):
        self.client = OpenTargetsClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_get_target_associations(self):
        """Test get_target_associations returns filtered associations with all fields."""
        associations = await self.client.get_target_data_associations(
            "ENSG00000112164", min_score=0.1
        )

        self.assertGreater(len(associations), 10)
        [gastroparesis] = [a for a in associations if a.disease_name == "gastroparesis"]
        # Verify all Association fields
        self.assertEqual(gastroparesis.disease_id, "EFO_1000948")
        self.assertEqual(gastroparesis.disease_name, "gastroparesis")
        self.assertGreater(gastroparesis.overall_score, 0.2)
        self.assertTrue(
            0.4 < gastroparesis.datatype_scores["genetic_association"] < 0.5
        )
        self.assertTrue(0.2 < gastroparesis.datatype_scores["literature"] < 0.3)
        self.assertIn("gastrointestinal disease", gastroparesis.therapeutic_areas)

    async def test_get_target_pathways(self):
        """Test get_target_pathways returns pathway data with all fields."""
        pathways = await self.client.get_target_data_pathways("ENSG00000113721")

        self.assertGreater(len(pathways), 5)
        [pdgf] = [p for p in pathways if p.pathway_name == "Signaling by PDGF"]
        # Verify all Pathway fields
        self.assertEqual(pdgf.pathway_id, "R-HSA-186797")
        self.assertEqual(pdgf.pathway_name, "Signaling by PDGF")
        self.assertEqual(pdgf.top_level_pathway, "Signal Transduction")

    async def test_get_target_interactions(self):
        """Test get_target_interactions returns interaction data with all fields."""
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        self.assertGreater(len(interactions), 10)
        plcg1 = next(
            i
            for i in interactions
            if i.interacting_target_symbol == "PLCG1" and i.source_database == "string"
        )
        # Verify all Interaction fields
        self.assertEqual(plcg1.interacting_target_id, "ENSG00000124181")
        self.assertEqual(plcg1.interacting_target_symbol, "PLCG1")
        self.assertGreater(plcg1.interaction_score, 0.99)
        self.assertEqual(plcg1.source_database, "string")
        self.assertEqual(plcg1.biological_role, "unspecified role")
        self.assertEqual(plcg1.evidence_count, 4)
        self.assertEqual(plcg1.interaction_type, "functional")

    async def test_interaction_type_string_is_functional(self):
        """Test STRING interaction with all Interaction fields verified."""
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        # Pick PLCG1 interaction from STRING - verify all fields
        plcg1_string = next(
            i
            for i in interactions
            if i.interacting_target_symbol == "PLCG1" and i.source_database == "string"
        )
        self.assertEqual(plcg1_string.interacting_target_id, "ENSG00000124181")
        self.assertEqual(plcg1_string.interacting_target_symbol, "PLCG1")
        self.assertGreater(plcg1_string.interaction_score, 0.99)
        self.assertEqual(plcg1_string.source_database, "string")
        self.assertEqual(plcg1_string.biological_role, "unspecified role")
        self.assertEqual(plcg1_string.evidence_count, 4)
        self.assertEqual(plcg1_string.interaction_type, "functional")

    async def test_interaction_type_intact_is_physical(self):
        """Test IntAct interaction with all Interaction fields verified."""
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        # Pick PLCG1 interaction from IntAct - verify all fields
        plcg1_intact = next(
            i
            for i in interactions
            if i.interacting_target_symbol == "PLCG1" and i.source_database == "intact"
        )
        self.assertEqual(plcg1_intact.interacting_target_id, "ENSG00000124181")
        self.assertEqual(plcg1_intact.interacting_target_symbol, "PLCG1")
        self.assertGreater(plcg1_intact.interaction_score, 0.6)
        self.assertEqual(plcg1_intact.source_database, "intact")
        self.assertEqual(plcg1_intact.biological_role, "unspecified role")
        self.assertGreater(plcg1_intact.evidence_count, 0)
        self.assertEqual(plcg1_intact.interaction_type, "physical")

    async def test_interaction_type_signor_is_signalling(self):
        """Test Signor interaction with all Interaction fields verified.

        Note: Signor data may not be currently available in Open Targets API.
        This test validates the mapping if Signor interactions are present.
        """
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        signor_interactions = [i for i in interactions if i.source_database == "signor"]
        for interaction in signor_interactions:
            self.assertEqual(interaction.interaction_type, "signalling")

    async def test_interaction_type_reactome_is_enzymatic(self):
        """Test Reactome interaction with all Interaction fields verified.

        Note: Reactome data may not be currently available in Open Targets API.
        This test validates the mapping if Reactome interactions are present.
        """
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        reactome_interactions = [
            i for i in interactions if i.source_database == "reactome"
        ]
        for interaction in reactome_interactions:
            self.assertEqual(interaction.interaction_type, "enzymatic")

    async def test_get_target_drug_summaries(self):
        """Test get_known_drugs returns drugs with all DrugSummary fields."""
        drug_summaries = await self.client.get_target_data_drug_summaries(
            "ENSG00000112164"
        )

        self.assertGreater(len(drug_summaries), 5)
        liraglutide = next(
            d
            for d in drug_summaries
            if d.drug_name == "LIRAGLUTIDE"
            and d.disease_name == "type 2 diabetes mellitus"
        )
        # Verify all DrugSummary fields
        self.assertEqual(liraglutide.drug_id, "CHEMBL4084119")
        self.assertEqual(liraglutide.drug_name, "LIRAGLUTIDE")
        self.assertEqual(liraglutide.disease_id, "MONDO_0005148")
        self.assertEqual(liraglutide.disease_name, "type 2 diabetes mellitus")
        self.assertEqual(liraglutide.phase, 4.0)
        self.assertIsNone(liraglutide.status)
        self.assertEqual(
            liraglutide.mechanism_of_action, "Glucagon-like peptide 1 receptor agonist"
        )
        self.assertEqual(liraglutide.clinical_trial_ids, [])

    async def test_get_target_expression(self):
        """Test get_target_expression returns tissue expression with all fields."""
        expressions = await self.client.get_target_data_tissue_expression(
            "ENSG00000163399"
        )

        self.assertGreater(len(expressions), 10)
        liver = next(e for e in expressions if e.tissue_name == "liver")
        # Verify all TissueExpression fields
        self.assertEqual(liver.tissue_id, "UBERON_0002107")
        self.assertEqual(liver.tissue_name, "liver")
        self.assertEqual(liver.tissue_anatomical_system, "endocrine system")
        # Verify all RNAExpression fields
        self.assertEqual(liver.rna.value, 11819.0)
        self.assertEqual(liver.rna.quantile, 5)
        self.assertEqual(liver.rna.unit, "TPM")
        # Verify all ProteinExpression fields
        self.assertEqual(liver.protein.level, 2)
        self.assertTrue(liver.protein.reliability)
        self.assertGreater(len(liver.protein.cell_types), 0)
        # Verify CellTypeExpression fields
        hepatocytes = next(
            ct for ct in liver.protein.cell_types if ct.name == "hepatocytes"
        )
        self.assertEqual(hepatocytes.name, "hepatocytes")
        self.assertEqual(hepatocytes.level, 1)
        self.assertTrue(hepatocytes.reliability)

    async def test_get_target_phenotypes(self):
        """Test get_target_phenotypes returns mouse phenotype with all fields."""
        phenotypes = await self.client.get_target_data_mouse_phenotypes(
            "ENSG00000112164"
        )

        self.assertGreater(len(phenotypes), 5)
        glucose = next(p for p in phenotypes if p.phenotype_id == "MP:0013279")
        # Verify all MousePhenotype fields
        self.assertEqual(glucose.phenotype_id, "MP:0013279")
        self.assertEqual(
            glucose.phenotype_label, "increased fasting circulating glucose level"
        )
        self.assertIn("homeostasis/metabolism phenotype", glucose.phenotype_categories)
        self.assertEqual(len(glucose.biological_models), 1)
        # Verify BiologicalModel fields
        [model] = glucose.biological_models
        self.assertEqual(model.allelic_composition, "Glp1r<tm1b(KOMP)Mbp> hom early")
        self.assertEqual(model.genetic_background, "C57BL/6NTac")
        self.assertEqual(model.literature, [])
        self.assertEqual(model.model_id, "")

    async def test_get_target_safety_liabilities(self):
        """Test get_target_data_safety_liabilities returns all SafetyLiability fields."""
        liabilities = await self.client.get_target_data_safety_liabilities(
            "ENSG00000163399"
        )

        self.assertGreater(len(liabilities), 5)

        arrhythmia = next(
            sl
            for sl in liabilities
            if sl.event == "cardiac arrhythmia" and sl.event_id == "EFO_0004269"
        )
        # Verify all SafetyLiability fields
        self.assertEqual(arrhythmia.event, "cardiac arrhythmia")
        self.assertEqual(arrhythmia.event_id, "EFO_0004269")
        self.assertEqual(arrhythmia.datasource, "Lynch et al. (2017)")
        self.assertEqual(arrhythmia.literature, "28216264")
        self.assertIsNone(arrhythmia.url)
        self.assertEqual(len(arrhythmia.effects), 1)
        # Verify SafetyEffect fields
        [effect] = arrhythmia.effects
        self.assertEqual(effect.direction, "Inhibition/Decrease/Downregulation")
        self.assertEqual(effect.dosing, "acute")

    async def test_get_target_genetic_constraints(self):
        """Test get_target_data_genetic_constraints returns all GeneticConstraint fields."""
        constraints = await self.client.get_target_data_genetic_constraints(
            "ENSG00000141736"
        )

        self.assertGreater(len(constraints), 0)
        lof_constraint = next(gc for gc in constraints if gc.constraint_type == "lof")
        # Verify all GeneticConstraint fields
        self.assertEqual(lof_constraint.constraint_type, "lof")
        self.assertTrue(0.41 < lof_constraint.oe < 0.42)
        self.assertTrue(0.33 < lof_constraint.oe_lower < 0.34)
        self.assertTrue(0.51 < lof_constraint.oe_upper < 0.52)
        self.assertTrue(0.06 < lof_constraint.score < 0.07)
        self.assertEqual(lof_constraint.upper_bin, 1)

    async def test_get_drug_indications(self):
        """Test get_drug_indications returns indication data."""
        indications = await self.client.get_drug_indications("semaglutide")

        self.assertGreater(len(indications), 5)
        [t2d] = [i for i in indications if i.disease_name == "type 2 diabetes mellitus"]
        self.assertEqual(t2d.disease_id, "MONDO_0005148")
        self.assertEqual(t2d.max_phase, 4.0)
        self.assertEqual(len(t2d.references), 4)

    async def test_get_disease_drugs(self):
        """Test get_disease_drugs returns drugs for a disease with all DrugSummary fields."""
        # Type 2 diabetes mellitus - should have semaglutide and other GLP-1 agonists
        drugs = await self.client.get_disease_drugs("MONDO_0005148")

        self.assertGreater(len(drugs), 10)
        semaglutide = next(
            d
            for d in drugs
            if d.drug_name == "SEMAGLUTIDE"
            and d.mechanism_of_action == "Glucagon-like peptide 1 receptor agonist"
        )
        # Verify all DrugSummary fields
        self.assertEqual(semaglutide.drug_id, "CHEMBL2108724")
        self.assertEqual(semaglutide.drug_name, "SEMAGLUTIDE")
        self.assertEqual(semaglutide.disease_id, "MONDO_0005148")
        self.assertEqual(semaglutide.disease_name, "type 2 diabetes mellitus")
        self.assertEqual(semaglutide.phase, 4.0)
        self.assertIsNone(semaglutide.status)
        self.assertEqual(
            semaglutide.mechanism_of_action, "Glucagon-like peptide 1 receptor agonist"
        )
        self.assertEqual(semaglutide.clinical_trial_ids, [])
