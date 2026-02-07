"""Integration tests for ClinicalTrialsClient."""

import unittest

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient


class TestClinicalTrialsClient(unittest.IsolatedAsyncioTestCase):
    """Integration tests for ClinicalTrialsClient."""

    async def asyncSetUp(self):
        self.client = ClinicalTrialsClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_get_landscape(self):
        """Test get_landscape returns competitive landscape for a condition."""
        landscape = await self.client.get_landscape("type 2 diabetes")

        # Verify ConditionLandscape.total_trial_count
        self.assertTrue(10000 < landscape.total_trial_count < 50000)

        # Verify ConditionLandscape.competitors
        self.assertEqual(len(landscape.competitors), 50)  # default top_n

        # Verify ConditionLandscape.phase_distribution
        self.assertTrue(100 < landscape.phase_distribution["Phase 3"] < 5000)
        self.assertTrue(100 < landscape.phase_distribution["Phase 2"] < 5000)

        # Verify ConditionLandscape.recent_starts - pick one and check fields
        self.assertTrue(len(landscape.recent_starts) > 0)
        recent = landscape.recent_starts[0]
        self.assertIn("nct_id", recent)
        self.assertIn("NCT", recent["nct_id"])

        # Find Novo Nordisk with Semaglutide - a known competitor in T2D
        novo_sema = next(
            c
            for c in landscape.competitors
            if "Novo Nordisk" in c.sponsor and "Semaglutide" in c.drug_name
        )

        # Verify all CompetitorEntry fields with exact values
        self.assertEqual(novo_sema.sponsor, "Novo Nordisk A/S")
        self.assertEqual(novo_sema.drug_name, "Semaglutide")
        self.assertEqual(novo_sema.drug_type, "Drug")
        self.assertEqual(novo_sema.max_phase, "Phase 4")
        self.assertTrue(10 < novo_sema.trial_count < 100)
        self.assertIn("COMPLETED", novo_sema.statuses)
        self.assertTrue(5000 < novo_sema.total_enrollment < 100000)
        self.assertTrue("2020-01-01" < novo_sema.most_recent_start < "2030-01-01")
