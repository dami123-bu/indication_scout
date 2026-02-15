"""Unit tests for Open Targets models."""

import pytest

from indication_scout.models.model_open_targets import (
    DrugData,
    DrugTarget,
    Indication,
)


# --- DrugData ---


@pytest.fixture
def sample_drug_data():
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
            DrugTarget(
                target_id="ENSG00000171016",
                target_symbol="PYGO1",
                mechanism_of_action="Pygo homolog 1 inhibitor",
                action_type="INHIBITOR",
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


def test_drug_data_contains_drug_targets(sample_drug_data):
    """DrugData.targets should contain DrugTarget objects with all fields."""
    assert len(sample_drug_data.targets) == 2

    # First target
    assert sample_drug_data.targets[0].target_id == "ENSG00000112164"
    assert sample_drug_data.targets[0].target_symbol == "GLP1R"
    assert (
        sample_drug_data.targets[0].mechanism_of_action == "GLP-1 receptor agonist"
    )
    assert sample_drug_data.targets[0].action_type == "AGONIST"

    # Second target
    assert sample_drug_data.targets[1].target_id == "ENSG00000171016"
    assert sample_drug_data.targets[1].target_symbol == "PYGO1"
    assert (
        sample_drug_data.targets[1].mechanism_of_action
        == "Pygo homolog 1 inhibitor"
    )
    assert sample_drug_data.targets[1].action_type == "INHIBITOR"


def test_drug_target_action_type_optional():
    """DrugTarget.action_type should be optional (None allowed)."""
    drug = DrugData(
        chembl_id="CHEMBL1234",
        name="TEST_DRUG",
        drug_type="Small molecule",
        is_approved=False,
        max_clinical_phase=2.0,
        targets=[
            DrugTarget(
                target_id="ENSG00000112164",
                target_symbol="GLP1R",
                mechanism_of_action="GLP-1 receptor modulator",
                action_type=None,
            ),
        ],
    )

    assert len(drug.targets) == 1
    assert drug.targets[0].target_id == "ENSG00000112164"
    assert drug.targets[0].target_symbol == "GLP1R"
    assert drug.targets[0].mechanism_of_action == "GLP-1 receptor modulator"
    assert drug.targets[0].action_type is None


def test_drug_data_empty_targets():
    """DrugData.targets should default to empty list."""
    drug = DrugData(
        chembl_id="CHEMBL9999",
        name="NO_TARGET_DRUG",
        drug_type="Small molecule",
        is_approved=False,
        max_clinical_phase=1.0,
    )

    assert drug.targets == []


def test_approved_disease_ids_returns_phase_4_only(sample_drug_data):
    """approved_disease_ids should return only phase 4+ disease IDs."""
    approved = sample_drug_data.approved_disease_ids
    assert approved == {"MONDO_0005148", "EFO_0001073"}
    assert "MONDO_0005155" not in approved


def test_investigated_disease_ids_returns_all(sample_drug_data):
    """investigated_disease_ids should return all disease IDs."""
    investigated = sample_drug_data.investigated_disease_ids
    assert investigated == {"MONDO_0005148", "EFO_0001073", "MONDO_0005155"}


def test_approved_disease_ids_empty_when_no_indications():
    """approved_disease_ids should be empty when no indications."""
    drug = DrugData(
        chembl_id="CHEMBL9999",
        name="NO_INDICATION_DRUG",
        drug_type="Small molecule",
        is_approved=False,
        max_clinical_phase=1.0,
        indications=[],
    )
    assert drug.approved_disease_ids == set()


def test_investigated_disease_ids_empty_when_no_indications():
    """investigated_disease_ids should be empty when no indications."""
    drug = DrugData(
        chembl_id="CHEMBL9999",
        name="NO_INDICATION_DRUG",
        drug_type="Small molecule",
        is_approved=False,
        max_clinical_phase=1.0,
        indications=[],
    )
    assert drug.investigated_disease_ids == set()
