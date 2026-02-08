"""Unit tests for PubMed models."""

import pytest
from pydantic import ValidationError

from indication_scout.models.model_pubmed import Publication


class TestPublication:
    """Tests for Publication model."""

    def test_publication_all_fields(self):
        """Publication should accept all fields with valid values."""
        pub = Publication(
            pmid="38472913",
            title="Semaglutide for NASH: Phase 3 Results",
            abstract="BACKGROUND: Nonalcoholic steatohepatitis... RESULTS: Resolution occurred in 59%...",
            journal="N Engl J Med",
            year=2024,
            publication_types=["Clinical Trial, Phase III", "Randomized Controlled Trial"],
            mesh_terms=["Non-alcoholic Fatty Liver Disease", "Glucagon-Like Peptide-1 Receptor"],
            doi="10.1056/NEJMoa2312345",
        )

        assert pub.pmid == "38472913"
        assert pub.title == "Semaglutide for NASH: Phase 3 Results"
        assert pub.abstract == "BACKGROUND: Nonalcoholic steatohepatitis... RESULTS: Resolution occurred in 59%..."
        assert pub.journal == "N Engl J Med"
        assert pub.year == 2024
        assert pub.publication_types == ["Clinical Trial, Phase III", "Randomized Controlled Trial"]
        assert pub.mesh_terms == ["Non-alcoholic Fatty Liver Disease", "Glucagon-Like Peptide-1 Receptor"]
        assert pub.doi == "10.1056/NEJMoa2312345"

    def test_publication_optional_fields_defaults(self):
        """Publication should use defaults for optional fields."""
        pub = Publication(
            pmid="12345678",
            title="Some Title",
            abstract="",
            journal="J Test Med",
            year=None,
            publication_types=[],
            mesh_terms=[],
        )

        assert pub.pmid == "12345678"
        assert pub.title == "Some Title"
        assert pub.abstract == ""
        assert pub.journal == "J Test Med"
        assert pub.year is None
        assert pub.publication_types == []
        assert pub.mesh_terms == []
        assert pub.doi is None

    def test_publication_year_none_for_epub(self):
        """Publication year should be None for epub-ahead-of-print articles."""
        pub = Publication(
            pmid="99999999",
            title="Epub Ahead of Print Article",
            abstract="This article has no publication year yet.",
            journal="Future Med",
            year=None,
            publication_types=["Journal Article"],
            mesh_terms=[],
        )

        assert pub.year is None

    def test_publication_empty_abstract(self):
        """Publication should accept empty abstract string."""
        pub = Publication(
            pmid="11111111",
            title="Article Without Abstract",
            abstract="",
            journal="Brief Rep",
            year=2023,
            publication_types=["Letter"],
            mesh_terms=[],
        )

        assert pub.abstract == ""

    @pytest.mark.parametrize(
        "pub_types,description",
        [
            (["Meta-Analysis"], "highest evidence tier"),
            (["Systematic Review"], "high evidence tier"),
            (["Randomized Controlled Trial"], "gold standard clinical evidence"),
            (["Clinical Trial, Phase III"], "late-stage efficacy data"),
            (["Clinical Trial, Phase II"], "dose-finding / early efficacy"),
            (["Clinical Trial, Phase I"], "safety / PK only"),
            (["Clinical Trial"], "unspecified phase"),
            (["Observational Study"], "real-world evidence"),
            (["Case Reports"], "anecdotal evidence"),
            (["Review"], "not primary evidence"),
            (["Letter"], "commentary"),
            (["Retracted Publication"], "evidence withdrawn"),
        ],
    )
    def test_publication_types_nlm_vocabulary(self, pub_types, description):
        """Publication should accept NLM controlled vocabulary publication types."""
        pub = Publication(
            pmid="00000001",
            title=f"Test {description}",
            abstract="Test abstract",
            journal="Test J",
            year=2024,
            publication_types=pub_types,
            mesh_terms=[],
        )

        assert pub.publication_types == pub_types

    def test_publication_multiple_publication_types(self):
        """Publication can have multiple publication types."""
        pub = Publication(
            pmid="22222222",
            title="RCT Phase III",
            abstract="A randomized controlled phase 3 trial.",
            journal="Lancet",
            year=2024,
            publication_types=["Clinical Trial, Phase III", "Randomized Controlled Trial", "Multicenter Study"],
            mesh_terms=["Diabetes Mellitus, Type 2"],
        )

        assert len(pub.publication_types) == 3
        assert "Randomized Controlled Trial" in pub.publication_types
        assert "Clinical Trial, Phase III" in pub.publication_types
        assert "Multicenter Study" in pub.publication_types

    def test_publication_multiple_mesh_terms(self):
        """Publication can have multiple MeSH terms."""
        pub = Publication(
            pmid="33333333",
            title="GLP-1 in Liver Disease",
            abstract="Study of GLP-1 agonists in liver disease.",
            journal="Hepatology",
            year=2024,
            publication_types=["Journal Article"],
            mesh_terms=[
                "Non-alcoholic Fatty Liver Disease",
                "Glucagon-Like Peptide-1 Receptor",
                "Liver Cirrhosis",
                "Humans",
            ],
        )

        assert len(pub.mesh_terms) == 4
        assert "Non-alcoholic Fatty Liver Disease" in pub.mesh_terms
        assert "Glucagon-Like Peptide-1 Receptor" in pub.mesh_terms

    @pytest.mark.parametrize(
        "missing_field",
        ["pmid", "title", "abstract", "journal", "publication_types", "mesh_terms"],
    )
    def test_publication_required_fields(self, missing_field):
        """Publication should require all non-optional fields."""
        data = {
            "pmid": "12345678",
            "title": "Test Title",
            "abstract": "Test abstract",
            "journal": "Test J",
            "year": 2024,
            "publication_types": [],
            "mesh_terms": [],
        }
        del data[missing_field]

        with pytest.raises(ValidationError):
            Publication(**data)

    def test_publication_year_required_but_nullable(self):
        """Publication year is required but can be None."""
        # year must be explicitly provided (even if None)
        data = {
            "pmid": "12345678",
            "title": "Test Title",
            "abstract": "Test abstract",
            "journal": "Test J",
            "publication_types": [],
            "mesh_terms": [],
        }

        with pytest.raises(ValidationError):
            Publication(**data)

        # But None is valid
        data["year"] = None
        pub = Publication(**data)
        assert pub.year is None
