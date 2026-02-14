"""Integration tests for OpenTargetsClient."""

import logging
import unittest

from indication_scout.data_sources.open_targets import OpenTargetsClient

logger = logging.getLogger(__name__)


class TestGetDrugData(unittest.IsolatedAsyncioTestCase):
    """Integration tests for get_drug method."""

    async def asyncSetUp(self):
        self.client = OpenTargetsClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_sildenafil_drug_data(self):
        """Test fetching drug data and indications for semaglutide."""
        drug = await self.client.get_drug("Semaglutide")
        indications = drug.indications
        match = [i for i in indications if "kidney" in i.disease_name.lower()]
        approved = [a for a in match if a.disease_id in drug.approved_disease_ids]
        logger.info(drug.indications)

    async def test_semaglutide_drug_data(self):
        """Test fetching drug data and indications for semaglutide."""
        drug = await self.client.get_drug("semaglutide")

        # DrugData top-level fields
        self.assertEqual(drug.chembl_id, "CHEMBL2108724")
        self.assertEqual(drug.name, "SEMAGLUTIDE")
        self.assertIn("NN-9535", drug.synonyms)
        self.assertIn("Ozempic", drug.trade_names)
        self.assertIn("Wegovy", drug.trade_names)
        self.assertEqual(drug.drug_type, "Protein")
        self.assertTrue(drug.is_approved)
        self.assertEqual(drug.max_clinical_phase, 4.0)
        self.assertEqual(drug.year_first_approved, 2017)
        self.assertGreater(len(drug.indications), 5)
        self.assertGreater(len(drug.targets), 0)
        self.assertGreater(len(drug.adverse_events), 5)
        self.assertTrue(38.5 < drug.adverse_events_critical_value < 38.6)
        self.assertEqual(len(drug.warnings), 1)

        # Indications - should include type 2 diabetes (approved)
        t2d_indication = next(
            (
                i
                for i in drug.indications
                if "type 2 diabetes" in i.disease_name.lower()
            ),
            None,
        )
        self.assertIsNotNone(t2d_indication)
        self.assertEqual(t2d_indication.max_phase, 4.0)

    async def test_semaglutide_drug_target(self):
        """Test DrugTarget fields for semaglutide's GLP1R target."""
        drug = await self.client.get_drug("semaglutide")
        [glp1r] = [t for t in drug.targets if t.target_symbol == "GLP1R"]

        self.assertEqual(glp1r.target_id, "ENSG00000112164")
        self.assertEqual(glp1r.target_symbol, "GLP1R")
        self.assertEqual(
            glp1r.mechanism_of_action, "Glucagon-like peptide 1 receptor agonist"
        )
        self.assertEqual(glp1r.action_type, "AGONIST")

    async def test_semaglutide_indication(self):
        """Test Indication fields for semaglutide's type 2 diabetes indication."""
        drug = await self.client.get_drug("semaglutide")
        [t2d] = [
            i for i in drug.indications if i.disease_name == "type 2 diabetes mellitus"
        ]

        self.assertEqual(t2d.disease_id, "MONDO_0005148")
        self.assertEqual(t2d.disease_name, "type 2 diabetes mellitus")
        self.assertEqual(t2d.max_phase, 4.0)
        self.assertEqual(len(t2d.references), 4)
        fda_ref = next(r for r in t2d.references if r["source"] == "FDA")
        self.assertIn("label/2017/209637lbl.pdf", fda_ref["ids"])

    async def test_trastuzumab_adverse_event(self):
        """Test AdverseEvent fields for trastuzumab."""
        drug = await self.client.get_drug("trastuzumab")
        adverse_event = next(
            ae for ae in drug.adverse_events if ae.name == "ejection fraction decreased"
        )

        self.assertEqual(adverse_event.name, "ejection fraction decreased")
        self.assertEqual(adverse_event.meddra_code, "10050528")
        self.assertEqual(adverse_event.count, 1124)
        self.assertTrue(2725.0 < adverse_event.log_likelihood_ratio < 2726.0)

    async def test_rofecoxib_drug_warning(self):
        """Test DrugWarning fields for rofecoxib (Vioxx) - a withdrawn drug with complete warning data."""
        drug = await self.client.get_drug("rofecoxib")
        self.assertGreater(len(drug.warnings), 5)

        # Find a specific warning with all fields populated
        warning = next(
            w
            for w in drug.warnings
            if w.toxicity_class == "cardiotoxicity"
            and w.efo_id == "EFO:0000612"
            and "serious cardiovascular events" in (w.description or "")
        )

        self.assertEqual(warning.warning_type, "Withdrawn")
        self.assertEqual(
            warning.description,
            "Increased risk of serious cardiovascular events, such as heart attacks and strokes",
        )
        self.assertEqual(warning.toxicity_class, "cardiotoxicity")
        self.assertEqual(warning.country, "Worldwide")
        self.assertEqual(warning.year, 2004)
        self.assertEqual(warning.efo_id, "EFO:0000612")

    async def test_metformin_drug_data(self):
        """Simple test for metformin - a different drug than others in the suite."""
        drug = await self.client.get_drug("metformin")

        self.assertEqual(drug.chembl_id, "CHEMBL1431")
        self.assertEqual(drug.name, "METFORMIN")
        self.assertEqual(drug.drug_type, "Small molecule")
        self.assertTrue(drug.is_approved)
        self.assertEqual(drug.max_clinical_phase, 4.0)
        self.assertEqual(drug.year_first_approved, 1995)


class TestGetDrugDataErrors(unittest.IsolatedAsyncioTestCase):
    """Tests for error handling in get_drug method."""

    async def asyncSetUp(self):
        self.client = OpenTargetsClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_nonexistent_drug_raises_error(self):
        """Test that a nonexistent drug raises DataSourceError."""
        from indication_scout.data_sources.base_client import DataSourceError

        with self.assertRaises(DataSourceError) as ctx:
            await self.client.get_drug("xyzzy_not_a_real_drug_12345")

        self.assertEqual(ctx.exception.source, "open_targets")
        self.assertIn("No drug found", str(ctx.exception))

    async def test_empty_drug_name_raises_error(self):
        """Test that empty drug name raises DataSourceError."""
        from indication_scout.data_sources.base_client import DataSourceError

        with self.assertRaises(DataSourceError) as ctx:
            await self.client.get_drug("")

        self.assertEqual(ctx.exception.source, "open_targets")

    async def test_special_characters_drug_name_raises_error(self):
        """Test that special characters in drug name raises DataSourceError."""
        from indication_scout.data_sources.base_client import DataSourceError

        with self.assertRaises(DataSourceError) as ctx:
            await self.client.get_drug("!!!@@@###$$$")

        self.assertEqual(ctx.exception.source, "open_targets")


class TestGetTargetDataErrors(unittest.IsolatedAsyncioTestCase):
    """Tests for error handling in get_target_data method."""

    async def asyncSetUp(self):
        self.client = OpenTargetsClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_nonexistent_target_raises_error(self):
        """Test that a nonexistent target ID raises DataSourceError."""
        from indication_scout.data_sources.base_client import DataSourceError

        with self.assertRaises(DataSourceError) as ctx:
            await self.client.get_target_data("ENSG99999999999")

        self.assertEqual(ctx.exception.source, "open_targets")

    async def test_invalid_target_format_raises_error(self):
        """Test that invalid target format raises DataSourceError."""
        from indication_scout.data_sources.base_client import DataSourceError

        with self.assertRaises(DataSourceError) as ctx:
            await self.client.get_target_data("not_an_ensembl_id")

        self.assertEqual(ctx.exception.source, "open_targets")

    async def test_empty_target_id_raises_error(self):
        """Test that empty target ID raises DataSourceError."""
        from indication_scout.data_sources.base_client import DataSourceError

        with self.assertRaises(DataSourceError) as ctx:
            await self.client.get_target_data("")

        self.assertEqual(ctx.exception.source, "open_targets")


class TestGetTargetData(unittest.IsolatedAsyncioTestCase):
    """Integration tests for get_target method."""

    async def asyncSetUp(self):
        self.client = OpenTargetsClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_glp1r_target_associations(self):
        """Test fetching associations for GLP1R target."""
        target = await self.client.get_target_data("ENSG00000112164")

        self.assertGreater(len(target.associations), 10)
        [assoc] = [a for a in target.associations if a.disease_name == "gastroparesis"]

        self.assertIsNotNone(assoc)
        self.assertTrue(
            assoc.disease_id.startswith("EFO_") or assoc.disease_id.startswith("MONDO_")
        )
        self.assertGreater(assoc.overall_score, 0.2)
        self.assertTrue(0.4 < assoc.datatype_scores["genetic_association"] < 0.5)
        self.assertTrue(0.2 < assoc.datatype_scores["literature"] < 0.3)
        self.assertIn("gastrointestinal disease", assoc.therapeutic_areas)

    async def test_glp1r_drug_summary(self):
        """Test DrugSummary fields for GLP1R target."""
        target = await self.client.get_target_data("ENSG00000112164")
        liraglutide = next(
            d
            for d in target.drug_summaries
            if d.drug_name == "LIRAGLUTIDE"
            and d.disease_name == "type 2 diabetes mellitus"
        )

        self.assertEqual(liraglutide.drug_id, "CHEMBL4084119")
        self.assertEqual(liraglutide.drug_name, "LIRAGLUTIDE")
        self.assertEqual(liraglutide.disease_id, "MONDO_0005148")
        self.assertEqual(liraglutide.disease_name, "type 2 diabetes mellitus")
        self.assertEqual(liraglutide.phase, 4.0)
        self.assertIsNone(liraglutide.status)
        self.assertEqual(
            liraglutide.mechanism_of_action, "Glucagon-like peptide 1 receptor agonist"
        )
        self.assertIsInstance(liraglutide.clinical_trial_ids, list)

    async def test_pdgfrb_target_pathway(self):
        """Test Pathway fields for PDGFRB target."""
        target = await self.client.get_target_data("ENSG00000113721")
        [pathway] = [
            p for p in target.pathways if p.pathway_name == "Signaling by PDGF"
        ]

        self.assertEqual(pathway.pathway_id, "R-HSA-186797")
        self.assertEqual(pathway.pathway_name, "Signaling by PDGF")
        self.assertEqual(pathway.top_level_pathway, "Signal Transduction")

    async def test_pdgfrb_target_interaction(self):
        """Test Interaction fields for PDGFRB target."""
        target = await self.client.get_target_data("ENSG00000113721")
        interaction = next(
            i
            for i in target.interactions
            if i.interacting_target_symbol == "PLCG1" and i.source_database == "string"
        )

        self.assertEqual(interaction.interacting_target_id, "ENSG00000124181")
        self.assertEqual(interaction.interacting_target_symbol, "PLCG1")
        self.assertGreater(interaction.interaction_score, 0.99)
        self.assertEqual(interaction.source_database, "string")
        self.assertEqual(interaction.biological_role, "unspecified role")
        self.assertEqual(interaction.evidence_count, 4)

    async def test_atp1a1_target_tissue_expression(self):
        """Test TissueExpression, RNAExpression, ProteinExpression, and CellTypeExpression fields."""
        target = await self.client.get_target_data("ENSG00000163399")
        expression = next(e for e in target.expressions if e.tissue_name == "liver")

        # TissueExpression fields
        self.assertEqual(expression.tissue_id, "UBERON_0002107")
        self.assertEqual(expression.tissue_name, "liver")
        self.assertEqual(expression.tissue_anatomical_system, "endocrine system")

        # RNAExpression fields
        self.assertEqual(expression.rna.value, 11819.0)
        self.assertEqual(expression.rna.quantile, 5)
        self.assertEqual(expression.rna.unit, "TPM")

        # ProteinExpression fields
        self.assertEqual(expression.protein.level, 2)
        self.assertTrue(expression.protein.reliability)
        self.assertGreater(len(expression.protein.cell_types), 0)

        # CellTypeExpression fields
        hepatocytes = next(
            ct for ct in expression.protein.cell_types if ct.name == "hepatocytes"
        )
        self.assertEqual(hepatocytes.name, "hepatocytes")
        self.assertEqual(hepatocytes.level, 1)
        self.assertTrue(hepatocytes.reliability)

    async def test_glp1r_target_mouse_phenotype(self):
        """Test MousePhenotype and BiologicalModel fields for GLP1R target."""
        target = await self.client.get_target_data("ENSG00000112164")
        phenotype = next(
            p for p in target.mouse_phenotypes if p.phenotype_id == "MP:0013279"
        )

        # MousePhenotype fields
        self.assertEqual(phenotype.phenotype_id, "MP:0013279")
        self.assertEqual(
            phenotype.phenotype_label, "increased fasting circulating glucose level"
        )
        self.assertIn(
            "homeostasis/metabolism phenotype", phenotype.phenotype_categories
        )
        self.assertEqual(len(phenotype.biological_models), 1)

        # BiologicalModel fields
        [model] = phenotype.biological_models
        self.assertEqual(model.allelic_composition, "Glp1r<tm1b(KOMP)Mbp> hom early")
        self.assertEqual(model.genetic_background, "C57BL/6NTac")

    async def test_erbb2_target_genetic_constraint(self):
        """Test GeneticConstraint fields for ERBB2 target."""
        target = await self.client.get_target_data("ENSG00000141736")
        lof_constraint = next(
            gc for gc in target.genetic_constraint if gc.constraint_type == "lof"
        )

        self.assertEqual(lof_constraint.constraint_type, "lof")
        self.assertTrue(0.41 < lof_constraint.oe < 0.42)
        self.assertTrue(0.33 < lof_constraint.oe_lower < 0.34)
        self.assertTrue(0.51 < lof_constraint.oe_upper < 0.52)
        self.assertTrue(0.06 < lof_constraint.score < 0.07)
        self.assertEqual(lof_constraint.upper_bin, 1)

    async def test_atp1a1_target_safety_liability(self):
        """Test SafetyLiability and SafetyEffect fields for ATP1A1 target."""
        target = await self.client.get_target_data("ENSG00000163399")
        self.assertGreater(len(target.safety_liabilities), 5)

        # Find a specific safety liability with effects
        liability = next(
            sl
            for sl in target.safety_liabilities
            if sl.event == "cardiac arrhythmia" and sl.event_id == "EFO_0004269"
        )

        # SafetyLiability fields
        self.assertEqual(liability.event, "cardiac arrhythmia")
        self.assertEqual(liability.event_id, "EFO_0004269")
        self.assertEqual(liability.datasource, "Lynch et al. (2017)")
        self.assertEqual(liability.literature, "28216264")
        self.assertIsNone(liability.url)
        self.assertEqual(len(liability.effects), 1)

        # SafetyEffect fields
        [effect] = liability.effects
        self.assertEqual(effect.direction, "Inhibition/Decrease/Downregulation")
        self.assertEqual(effect.dosing, "acute")
