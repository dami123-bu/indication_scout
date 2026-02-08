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
        """Test get_landscape returns competitive landscape for a condition.

        get_landscape filters to intervention_type in ("Drug", "Biological") only,
        groups by sponsor + drug, and ranks by phase (desc) then enrollment (desc).
        """
        # Use a smaller, more stable condition for predictable results
        landscape = await self.client.get_landscape("gastroparesis", top_n=10)

        # Verify ConditionLandscape.total_trial_count - gastroparesis has ~300 trials
        self.assertTrue(80 < landscape.total_trial_count < 150)

        # Verify ConditionLandscape.competitors - requested top_n=10
        self.assertEqual(len(landscape.competitors), 10)

        # Verify ConditionLandscape.phase_distribution
        self.assertTrue(30 < landscape.phase_distribution["Phase 2"] < 100)
        self.assertTrue(5 < landscape.phase_distribution["Phase 3"] < 50)
        self.assertTrue(5 < landscape.phase_distribution["Phase 4"] < 30)

        # Verify ConditionLandscape.recent_starts - check dict structure
        self.assertTrue(len(landscape.recent_starts) >= 1)
        recent = landscape.recent_starts[0]
        self.assertEqual(set(recent.keys()), {"nct_id", "sponsor", "drug", "phase"})

        # Find Chinese University of Hong Kong with Esomeprazole - top ranked competitor
        [cuhk] = [
            c
            for c in landscape.competitors
            if c.sponsor == "Chinese University of Hong Kong"
            and c.drug_name == "Esomeprazole"
        ]

        # Verify all CompetitorEntry fields
        self.assertEqual(cuhk.sponsor, "Chinese University of Hong Kong")
        self.assertEqual(cuhk.drug_name, "Esomeprazole")
        self.assertEqual(cuhk.drug_type, "Drug")
        self.assertEqual(cuhk.max_phase, "Phase 4")
        self.assertEqual(cuhk.trial_count, 1)
        self.assertEqual(cuhk.statuses, {"COMPLETED"})
        self.assertEqual(cuhk.total_enrollment, 155)
        self.assertEqual(cuhk.most_recent_start, "2009-12-03")

    async def test_search_trials(self):
        """Test search_trials returns trials for a drug-condition pair."""
        # Search for trastuzumab + breast cancer, find a specific known trial
        trials = await self.client.search_trials(
            drug="trastuzumab",
            condition="breast cancer",
            max_results=50,
            phase_filter="PHASE4",
        )

        # Find NCT00127933 - XeNA Study (Roche, includes Herceptin)
        [xena] = [t for t in trials if t.nct_id == "NCT00127933"]

        # Verify all Trial fields with exact values
        self.assertEqual(xena.nct_id, "NCT00127933")
        self.assertEqual(
            xena.title,
            "XeNA Study - A Study of Xeloda (Capecitabine) in Patients With Invasive Breast Cancer",
        )
        self.assertEqual(xena.phase, "Phase 4")
        self.assertEqual(xena.overall_status, "COMPLETED")
        self.assertIsNone(xena.why_stopped)
        self.assertEqual(xena.conditions, ["Breast Cancer"])
        self.assertEqual(xena.sponsor, "Hoffmann-La Roche")
        self.assertEqual(xena.collaborators, [])
        self.assertEqual(xena.enrollment, 157)
        self.assertEqual(xena.start_date, "2005-08")
        self.assertEqual(xena.completion_date, "2009-04")
        self.assertEqual(xena.study_type, "INTERVENTIONAL")
        self.assertEqual(xena.results_posted, True)
        self.assertEqual(xena.references, [])

        # Verify interventions - trial has 5 interventions including Herceptin
        self.assertEqual(len(xena.interventions), 5)
        [herceptin] = [
            i
            for i in xena.interventions
            if i.intervention_name == "Herceptin (HER2-neu positive patients only)"
        ]
        self.assertEqual(herceptin.intervention_type, "Drug")

        # Verify primary_outcomes
        self.assertEqual(len(xena.primary_outcomes), 1)
        self.assertEqual(
            xena.primary_outcomes[0].measure,
            "Percentage of Participants Assessed for Pathological Complete Response (pCR) Plus Near Complete (npCR) in Primary Breast Tumor at Time of Definitive Surgery",
        )
        self.assertEqual(
            xena.primary_outcomes[0].time_frame,
            "at the time of definitive surgery; after four 3-week cycles (3-4 months)",
        )

    async def test_search_trials_phase_filter(self):
        """Test that phase_filter only returns trials matching the specified phase."""
        # Search for Phase 3 trials only
        trials = await self.client.search_trials(
            drug="semaglutide",
            condition="diabetes",
            max_results=20,
            phase_filter="PHASE3",
        )

        # All returned trials must be Phase 3
        self.assertTrue(
            len(trials) >= 5
        )  # semaglutide + diabetes has many Phase 3 trials
        for trial in trials:
            self.assertIn("Phase 3", trial.phase)

    async def test_get_terminated(self):
        """Test get_terminated returns terminated trials for a query.

        get_terminated filters to status in (TERMINATED, WITHDRAWN, SUSPENDED) only.
        It classifies stop reasons using keyword matching into categories:
        safety, efficacy, business, enrollment, other, unknown.
        """
        trials = await self.client.get_terminated("semaglutide", max_results=20)

        # Find NCT04012255 - Novo Nordisk semaglutide pen-injector trial
        [novo_trial] = [t for t in trials if t.nct_id == "NCT04012255"]

        # Verify all TerminatedTrial fields with exact values
        self.assertEqual(novo_trial.nct_id, "NCT04012255")
        self.assertEqual(
            novo_trial.title,
            "A Research Study to Compare Two Forms of Semaglutide in Two Different Pen-injectors in People With Overweight or Obesity",
        )
        self.assertEqual(
            novo_trial.drug_name, "Semaglutide (administered by DV3396 pen)"
        )
        self.assertEqual(novo_trial.condition, "Overweight")
        self.assertEqual(novo_trial.phase, "Phase 1")
        self.assertEqual(
            novo_trial.why_stopped, "The trial was terminated for strategic reasons."
        )
        self.assertEqual(novo_trial.stop_category, "business")
        self.assertEqual(novo_trial.enrollment, 29)
        self.assertEqual(novo_trial.sponsor, "Novo Nordisk A/S")
        self.assertEqual(novo_trial.start_date, "2019-07-15")
        self.assertEqual(novo_trial.termination_date, "2019-08-30")
        self.assertEqual(novo_trial.references, [])

    async def test_detect_whitespace(self):
        """Test detect_whitespace identifies unexplored drug-condition pairs.

        When is_whitespace=True, returns condition_drugs (other drugs being tested
        for this condition) ranked by phase (desc) then active status, deduplicated
        by drug_name.
        When is_whitespace=False, condition_drugs is empty.
        """
        # Tirzepatide + Huntington disease = whitespace (no exact matches)
        result = await self.client.detect_whitespace(
            "tirzepatide", "Huntington disease"
        )

        # Verify WhitespaceResult fields
        self.assertEqual(result.is_whitespace, True)
        self.assertEqual(result.exact_match_count, 0)
        self.assertTrue(150 < result.drug_only_trials < 300)
        self.assertTrue(200 < result.condition_only_trials < 400)

        # Verify condition_drugs
        self.assertTrue(40 < len(result.condition_drugs) < 60)

        # Verify deduplication: all drug_names should be unique
        drug_names = [cd.drug_name for cd in result.condition_drugs]
        self.assertEqual(len(drug_names), len(set(drug_names)))

        # Verify known drugs are found (Memantine and Tetrabenazine are Phase 4 HD drugs)
        self.assertIn("Memantine", drug_names)
        self.assertIn("Tetrabenazine", drug_names)

        # Verify ranking: first drugs should be Phase 4 (highest phase)
        self.assertEqual(result.condition_drugs[0].phase, "Phase 4")
        self.assertEqual(result.condition_drugs[1].phase, "Phase 4")
        self.assertEqual(result.condition_drugs[2].phase, "Phase 4")

        # Find Memantine (Phase 4 completed trial) and verify all ConditionDrug fields
        [memantine] = [
            cd for cd in result.condition_drugs if cd.drug_name == "Memantine"
        ]
        self.assertEqual(memantine.nct_id, "NCT00652457")
        self.assertEqual(memantine.drug_name, "Memantine")
        self.assertEqual(memantine.condition, "Huntington's Disease")
        self.assertEqual(memantine.phase, "Phase 4")
        self.assertEqual(memantine.status, "COMPLETED")

        # Verify Phase 2+ filter: no Phase 1 or Early Phase 1 trials
        for cd in result.condition_drugs:
            self.assertNotIn("Phase 1", cd.phase)
            self.assertNotIn("Early Phase 1", cd.phase)

    async def test_detect_whitespace_not_whitespace(self):
        """Test detect_whitespace when exact matches exist (is_whitespace=False).

        When trials exist for the drug-condition pair, is_whitespace=False
        and condition_drugs is empty (no need to show competitors).
        """
        # Semaglutide + diabetes = NOT whitespace (many exact matches)
        result = await self.client.detect_whitespace("semaglutide", "diabetes")

        # Verify WhitespaceResult fields for non-whitespace case
        self.assertEqual(result.is_whitespace, False)
        self.assertTrue(result.exact_match_count >= 10)  # semaglutide + diabetes has many trials
        self.assertTrue(200 < result.drug_only_trials < 600)
        self.assertTrue(10000 < result.condition_only_trials < 100000)

        # condition_drugs should be empty when not whitespace
        self.assertEqual(result.condition_drugs, [])
