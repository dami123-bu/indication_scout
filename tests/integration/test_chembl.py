"""Integration tests for ChEMBLClient — hits the live ChEMBL API."""

import pytest

from indication_scout.models.model_chembl import MoleculeData


@pytest.mark.parametrize(
    "chembl_id, expected_atc, expected_type, expected_max_phase, expected_black_box, expected_first_approval, expected_oral",
    [
        ("CHEMBL894", ["N06AX12"], "Small molecule", "4.0", 1, 1985, True),
        ("CHEMBL2108724", ["A10BJ06"], "Protein", "4.0", 1, 2017, True),
    ],
)
async def test_get_molecule(
    chembl_client,
    chembl_id,
    expected_atc,
    expected_type,
    expected_max_phase,
    expected_black_box,
    expected_first_approval,
    expected_oral,
):
    result = await chembl_client.get_molecule(chembl_id)

    assert isinstance(result, MoleculeData)
    assert result.molecule_chembl_id == chembl_id
    assert result.molecule_type == expected_type
    assert result.max_phase == expected_max_phase
    assert result.atc_classifications == expected_atc
    assert result.black_box_warning == expected_black_box
    assert result.first_approval == expected_first_approval
    assert result.oral == expected_oral
