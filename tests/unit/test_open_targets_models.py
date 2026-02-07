"""Unit tests for Open Targets models."""

import pytest

from indication_scout.models.open_targets import (
    DrugData,
    DrugEvaluation,
    DrugTarget,
    Indication,
    TargetData,
    TargetNotFoundError,
)


class TestTargetNotFoundError:
    """Tests for TargetNotFoundError exception."""

    def test_error_stores_target_id(self):
        """Error should store the target_id that was not found."""
        error = TargetNotFoundError("ENSG00000112164")
        assert error.target_id == "ENSG00000112164"

    def test_error_message_includes_target_id(self):
        """Error message should include the target_id."""
        error = TargetNotFoundError("ENSG00000112164")
        assert "ENSG00000112164" in str(error)


class TestDrugEvaluation:
    """Tests for DrugEvaluation model."""

    @pytest.fixture
    def sample_drug_data(self):
        """Create a sample DrugData for testing."""
        return DrugData(
            chembl_id="CHEMBL2108724",
            name="SEMAGLUTIDE",
            drug_type="Protein",
            is_approved=True,
            max_clinical_phase=4.0,
            targets=[
                DrugTarget(
                    target_id="ENSG00000112164",
                    target_symbol="GLP1R",
                    mechanism_of_action="GLP-1 receptor agonist",
                    action_type="AGONIST",
                ),
            ],
            indications=[
                Indication(
                    disease_id="MONDO_0005148",
                    disease_name="type 2 diabetes mellitus",
                    max_phase=4.0,
                ),
                Indication(
                    disease_id="EFO_0001073",
                    disease_name="obesity",
                    max_phase=4.0,
                ),
                Indication(
                    disease_id="MONDO_0005155",
                    disease_name="non-alcoholic steatohepatitis",
                    max_phase=3.0,
                ),
            ],
        )

    @pytest.fixture
    def sample_target_data(self):
        """Create a sample TargetData for testing."""
        return TargetData(
            target_id="ENSG00000112164",
            symbol="GLP1R",
            name="glucagon like peptide 1 receptor",
        )

    @pytest.fixture
    def sample_evaluation(self, sample_drug_data, sample_target_data):
        """Create a sample DrugEvaluation for testing."""
        return DrugEvaluation(
            drug=sample_drug_data,
            targets={"ENSG00000112164": sample_target_data},
        )

    def test_get_target_returns_target(self, sample_evaluation):
        """get_target should return the TargetData for a valid target_id."""
        target = sample_evaluation.get_target("ENSG00000112164")
        assert target.symbol == "GLP1R"
        assert target.target_id == "ENSG00000112164"

    def test_get_target_raises_for_unknown_id(self, sample_evaluation):
        """get_target should raise TargetNotFoundError for unknown target_id."""
        with pytest.raises(TargetNotFoundError) as exc_info:
            sample_evaluation.get_target("ENSG00000999999")
        assert exc_info.value.target_id == "ENSG00000999999"

    def test_primary_target_returns_first_target(self, sample_evaluation):
        """primary_target should return the first target in drug.targets."""
        primary = sample_evaluation.primary_target
        assert primary is not None
        assert primary.symbol == "GLP1R"

    def test_primary_target_returns_none_when_no_targets(self, sample_drug_data):
        """primary_target should return None when drug has no targets."""
        sample_drug_data.targets = []
        evaluation = DrugEvaluation(
            drug=sample_drug_data,
            targets={},
        )
        assert evaluation.primary_target is None

    def test_approved_disease_ids_returns_phase_4_only(self, sample_evaluation):
        """approved_disease_ids should return only phase 4+ disease IDs."""
        approved = sample_evaluation.approved_disease_ids
        assert approved == {"MONDO_0005148", "EFO_0001073"}
        assert "MONDO_0005155" not in approved

    def test_investigated_disease_ids_returns_all(self, sample_evaluation):
        """investigated_disease_ids should return all disease IDs."""
        investigated = sample_evaluation.investigated_disease_ids
        assert investigated == {"MONDO_0005148", "EFO_0001073", "MONDO_0005155"}

    def test_approved_disease_ids_empty_when_no_indications(self, sample_drug_data):
        """approved_disease_ids should be empty when no indications."""
        sample_drug_data.indications = []
        evaluation = DrugEvaluation(
            drug=sample_drug_data,
            targets={},
        )
        assert evaluation.approved_disease_ids == set()

    def test_investigated_disease_ids_empty_when_no_indications(self, sample_drug_data):
        """investigated_disease_ids should be empty when no indications."""
        sample_drug_data.indications = []
        evaluation = DrugEvaluation(
            drug=sample_drug_data,
            targets={},
        )
        assert evaluation.investigated_disease_ids == set()
