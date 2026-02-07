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
        """Test get_target_associations returns filtered associations."""
        associations = await self.client.get_target_data_associations(
            "ENSG00000112164", min_score=0.1
        )

        self.assertGreater(len(associations), 10)
        [gastroparesis] = [a for a in associations if a.disease_name == "gastroparesis"]
        self.assertGreater(gastroparesis.overall_score, 0.2)
        self.assertIn("gastrointestinal disease", gastroparesis.therapeutic_areas)

    async def test_get_target_pathways(self):
        """Test get_target_pathways returns pathway data."""
        pathways = await self.client.get_target_data_pathways("ENSG00000113721")

        self.assertGreater(len(pathways), 5)
        [pdgf] = [p for p in pathways if p.pathway_name == "Signaling by PDGF"]
        self.assertEqual(pdgf.pathway_id, "R-HSA-186797")
        self.assertEqual(pdgf.top_level_pathway, "Signal Transduction")

    async def test_get_target_interactions(self):
        """Test get_target_interactions returns interaction data."""
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        self.assertGreater(len(interactions), 10)
        plcg1 = next(
            i
            for i in interactions
            if i.interacting_target_symbol == "PLCG1" and i.source_database == "string"
        )
        self.assertEqual(plcg1.interacting_target_id, "ENSG00000124181")
        self.assertGreater(plcg1.interaction_score, 0.99)
        self.assertEqual(plcg1.evidence_count, 4)

    async def test_interaction_type_string_is_functional(self):
        """Test that STRING database interactions have interaction_type='functional'."""
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        string_interactions = [i for i in interactions if i.source_database == "string"]
        self.assertGreater(len(string_interactions), 0)
        for interaction in string_interactions:
            self.assertEqual(interaction.interaction_type, "functional")

    async def test_interaction_type_intact_is_physical(self):
        """Test that IntAct database interactions have interaction_type='physical'."""
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        intact_interactions = [i for i in interactions if i.source_database == "intact"]
        self.assertGreater(len(intact_interactions), 0)
        for interaction in intact_interactions:
            self.assertEqual(interaction.interaction_type, "physical")

    async def test_interaction_type_signor_is_signalling(self):
        """Test that Signor database interactions have interaction_type='signalling'.

        Note: Signor data may not be currently available in Open Targets API.
        This test validates the mapping if Signor interactions are present.
        """
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        signor_interactions = [i for i in interactions if i.source_database == "signor"]
        # Signor may not be available - test the mapping only if data exists
        for interaction in signor_interactions:
            self.assertEqual(interaction.interaction_type, "signalling")

    async def test_interaction_type_reactome_is_enzymatic(self):
        """Test that Reactome database interactions have interaction_type='enzymatic'.

        Note: Reactome data may not be currently available in Open Targets API.
        This test validates the mapping if Reactome interactions are present.
        """
        interactions = await self.client.get_target_data_interactions("ENSG00000113721")

        reactome_interactions = [
            i for i in interactions if i.source_database == "reactome"
        ]
        # Reactome may not be available - test the mapping only if data exists
        for interaction in reactome_interactions:
            self.assertEqual(interaction.interaction_type, "enzymatic")

    async def test_get_known_drugs(self):
        """Test get_known_drugs returns drugs targeting the same protein."""
        known_drugs = await self.client.get_target_data_known_drugs("ENSG00000112164")

        self.assertGreater(len(known_drugs), 5)
        liraglutide = next(
            d
            for d in known_drugs
            if d.drug_name == "LIRAGLUTIDE"
            and d.disease_name == "type 2 diabetes mellitus"
        )
        self.assertEqual(liraglutide.drug_id, "CHEMBL4084119")
        self.assertEqual(liraglutide.phase, 4.0)
        self.assertEqual(
            liraglutide.mechanism_of_action, "Glucagon-like peptide 1 receptor agonist"
        )

    async def test_get_target_expression(self):
        """Test get_target_expression returns tissue expression data."""
        expressions = await self.client.get_target_data_tissue_expression("ENSG00000163399")

        self.assertGreater(len(expressions), 10)
        liver = next(e for e in expressions if e.tissue_name == "liver")
        self.assertEqual(liver.tissue_id, "UBERON_0002107")
        self.assertEqual(liver.rna.value, 11819.0)
        self.assertEqual(liver.protein.level, 2)

    async def test_get_target_phenotypes(self):
        """Test get_target_phenotypes returns mouse phenotype data."""
        phenotypes = await self.client.get_target_data_mouse_phenotypes("ENSG00000112164")

        self.assertGreater(len(phenotypes), 5)
        glucose = next(p for p in phenotypes if p.phenotype_id == "MP:0013279")
        self.assertEqual(
            glucose.phenotype_label, "increased fasting circulating glucose level"
        )
        self.assertIn("homeostasis/metabolism phenotype", glucose.phenotype_categories)

    async def test_get_target_safety_liabilities(self):
        """Test get_target_data_safety_liabilities returns safety liability data."""
        liabilities = await self.client.get_target_data_safety_liabilities(
            "ENSG00000163399"
        )

        self.assertGreater(len(liabilities), 5)

        arrhythmia = next(
            sl
            for sl in liabilities
            if sl.event == "cardiac arrhythmia" and sl.event_id == "EFO_0004269"
        )
        self.assertEqual(arrhythmia.datasource, "Lynch et al. (2017)")

    async def test_get_target_genetic_constraints(self):
        """Test get_target_data_genetic_constraints returns genetic constraint data."""
        constraints = await self.client.get_target_data_genetic_constraints(
            "ENSG00000141736"
        )

        self.assertGreater(len(constraints), 0)
        lof_constraint = next(
            gc for gc in constraints if gc.constraint_type == "lof"
        )
        self.assertEqual(lof_constraint.constraint_type, "lof")
        self.assertTrue(0.41 < lof_constraint.oe < 0.42)

    async def test_get_drug_indications(self):
        """Test get_drug_indications returns indication data."""
        indications = await self.client.get_drug_indications("semaglutide")

        self.assertGreater(len(indications), 5)
        [t2d] = [i for i in indications if i.disease_name == "type 2 diabetes mellitus"]
        self.assertEqual(t2d.disease_id, "MONDO_0005148")
        self.assertEqual(t2d.max_phase, 4.0)
        self.assertEqual(len(t2d.references), 4)