"""Integration tests for PubMedClient."""

import os
import unittest

from dotenv import load_dotenv

from indication_scout.data_sources.pubmed import PubMedClient

load_dotenv()


class TestPubMedClient(unittest.IsolatedAsyncioTestCase):
    """Integration tests for PubMedClient."""

    async def asyncSetUp(self):
        api_key = os.getenv("NCBI_API_KEY")
        self.client = PubMedClient(api_key=api_key)

    async def asyncTearDown(self):
        await self.client.close()

    async def test_fetch_by_pmids_single(self):
        """Test fetch_by_pmids returns a single publication with all fields.

        PMID 33567185 is the STEP 1 trial: semaglutide for obesity (NEJM 2021).
        """
        publications = await self.client.fetch_by_pmids(["33567185"])

        self.assertEqual(len(publications), 1)
        pub = publications[0]

        # Verify all Publication fields
        self.assertEqual(pub.pmid, "33567185")
        self.assertEqual(
            pub.title,
            "Once-Weekly Semaglutide in Adults with Overweight or Obesity."
        )
        self.assertEqual(pub.journal, "N Engl J Med")
        self.assertEqual(pub.year, 2021)
        self.assertEqual(pub.doi, "10.1056/NEJMoa2032183")

        # Verify publication types
        self.assertIn("Randomized Controlled Trial", pub.publication_types)
        self.assertIn("Multicenter Study", pub.publication_types)

        # Verify MeSH terms include obesity-related terms
        self.assertIn("Anti-Obesity Agents", pub.mesh_terms)
        self.assertIn("Body Mass Index", pub.mesh_terms)

        # Verify abstract contains expected content
        self.assertIn("semaglutide", pub.abstract.lower())
        self.assertIn("weight", pub.abstract.lower())
        self.assertIn("obesity is a global health challenge", pub.abstract.lower())


    async def test_fetch_by_pmids_multiple(self):
        """Test fetch_by_pmids returns multiple publications.

        33567185 - semaglutide STEP 1 (NEJM 2021)
        35658024 - tirzepatide SURMOUNT-1 (NEJM 2022)
        """
        pmids = ["33567185", "35658024"]
        publications = await self.client.fetch_by_pmids(pmids)

        # Should return both publications
        self.assertEqual(len(publications), 2)

        # Verify all PMIDs are present (order may differ)
        returned_pmids = {p.pmid for p in publications}
        self.assertEqual(returned_pmids, {"33567185", "35658024"})

        # Find and verify STEP 1 publication
        [step1] = [p for p in publications if p.pmid == "33567185"]
        self.assertEqual(step1.title, "Once-Weekly Semaglutide in Adults with Overweight or Obesity.")
        self.assertEqual(step1.journal, "N Engl J Med")
        self.assertEqual(step1.year, 2021)
        self.assertIn("Randomized Controlled Trial", step1.publication_types)
        self.assertEqual('10.1056/NEJMoa2032183',step1.doi)
        expected_mesh_terms=('Anti-Obesity Agents', 'Body Composition', 'Healthy Lifestyle', 'Glucagon-Like Peptide 1')
        for term in expected_mesh_terms:
            self.assertIn(term, step1.mesh_terms)

        # Find and verify SURMOUNT-1 publication
        [surmount1] = [p for p in publications if p.pmid == "35658024"]
        self.assertEqual(surmount1.title, "Tirzepatide Once Weekly for the Treatment of Obesity.")
        self.assertEqual(surmount1.journal, "N Engl J Med")
        self.assertEqual(surmount1.year, 2022)
        self.assertIn("Clinical Trial, Phase III", surmount1.publication_types)
        self.assertEqual("10.1056/NEJMoa2206038", surmount1.doi)
        surmount1_mesh_terms = ("Anti-Obesity Agents", "Double-Blind Method", "Gastric Inhibitory Polypeptide")
        for term in surmount1_mesh_terms:
            self.assertIn(term, surmount1.mesh_terms)

    async def test_fetch_by_pmids_empty_list(self):
        """Test fetch_by_pmids returns empty list for empty input."""
        publications = await self.client.fetch_by_pmids([])

        self.assertEqual(publications, [])

    async def test_fetch_by_pmids_deduplicates(self):
        """Test fetch_by_pmids deduplicates input PMIDs."""
        pmids = ["33567185", "33567185", "33567185"]
        publications = await self.client.fetch_by_pmids(pmids)

        # Should return only 1 publication despite 3 duplicate inputs
        self.assertEqual(len(publications), 1)
        self.assertEqual(publications[0].pmid, "33567185")

    async def test_fetch_by_pmids_skips_invalid(self):
        """Test fetch_by_pmids skips invalid PMIDs silently."""
        pmids = ["33567185", "00000000", "99999999999"]
        publications = await self.client.fetch_by_pmids(pmids)

        # Should return only the valid publication
        self.assertEqual(len(publications), 1)
        self.assertEqual(publications[0].pmid, "33567185")

    async def test_fetch_by_pmids_clinical_trial_phase3(self):
        """Test fetch_by_pmids parses Phase III clinical trial publication types.

        PMID 35658024 is the SURMOUNT-1 trial: tirzepatide for obesity (NEJM 2022).
        """
        publications = await self.client.fetch_by_pmids(["35658024"])

        self.assertEqual(len(publications), 1)
        pub = publications[0]

        self.assertEqual(pub.pmid, "35658024")
        self.assertEqual(
            pub.title,
            "Tirzepatide Once Weekly for the Treatment of Obesity."
        )
        self.assertEqual(pub.journal, "N Engl J Med")
        self.assertEqual(pub.year, 2022)

        # Verify clinical trial publication types
        self.assertIn("Randomized Controlled Trial", pub.publication_types)
        self.assertIn("Clinical Trial, Phase III", pub.publication_types)

        # Verify obesity-related MeSH terms
        self.assertIn("Anti-Obesity Agents", pub.mesh_terms)
        self.assertIn("Double-Blind Method", pub.mesh_terms)
