"""Unit tests for ChEMBLClient.get_molecule()."""

from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_chembl import MoleculeData

CHEMBL894_FIXTURE = {
    "molecule_chembl_id": "CHEMBL894",
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": ["N06AX12"],
    "black_box_warning": 1,
    "first_approval": 1985,
    "oral": True,
}

CHEMBL2108724_FIXTURE = {
    "molecule_chembl_id": "CHEMBL2108724",
    "molecule_type": "Protein",
    "max_phase": "4.0",
    "atc_classifications": ["A10BJ06"],
    "black_box_warning": 1,
    "first_approval": 2017,
    "oral": True,
}

# Sildenafil
CHEMBL192_FIXTURE = {
    "molecule_chembl_id": "CHEMBL192",
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": ["G04BE03"],
    "black_box_warning": 0,
    "first_approval": 1998,
    "oral": True,
}

# Rituximab
CHEMBL1201585_FIXTURE = {
    "molecule_chembl_id": "CHEMBL1201585",
    "molecule_type": "Antibody",
    "max_phase": "4.0",
    "atc_classifications": ["L01FD01"],
    "black_box_warning": 1,
    "first_approval": 1998,
    "oral": False,
}

# Compound with null max_phase and no ATC codes
CHEMBL426_FIXTURE = {
    "molecule_chembl_id": "CHEMBL426",
    "molecule_type": "Small molecule",
    "max_phase": None,
    "atc_classifications": [],
    "black_box_warning": 0,
    "first_approval": None,
    "oral": False,
}


@pytest.mark.parametrize(
    "chembl_id, fixture, expected_atc, expected_type, expected_max_phase, expected_black_box, expected_first_approval, expected_oral",
    [
        ("CHEMBL894", CHEMBL894_FIXTURE, ["N06AX12"], "Small molecule", "4.0", 1, 1985, True),
        ("CHEMBL2108724", CHEMBL2108724_FIXTURE, ["A10BJ06"], "Protein", "4.0", 1, 2017, True),
        ("CHEMBL192", CHEMBL192_FIXTURE, ["G04BE03"], "Small molecule", "4.0", 0, 1998, True),
        ("CHEMBL1201585", CHEMBL1201585_FIXTURE, ["L01FD01"], "Antibody", "4.0", 1, 1998, False),
        ("CHEMBL426", CHEMBL426_FIXTURE, [], "Small molecule", None, 0, None, False),
    ],
)
async def test_get_molecule(
    chembl_id,
    fixture,
    expected_atc,
    expected_type,
    expected_max_phase,
    expected_black_box,
    expected_first_approval,
    expected_oral,
):
    client = ChEMBLClient()
    with patch.object(client, "_rest_get", new=AsyncMock(return_value=fixture)):
        result = await client.get_molecule(chembl_id)

    assert isinstance(result, MoleculeData)
    assert result.molecule_chembl_id == chembl_id
    assert result.molecule_type == expected_type
    assert result.max_phase == expected_max_phase
    assert result.atc_classifications == expected_atc
    assert result.black_box_warning == expected_black_box
    assert result.first_approval == expected_first_approval
    assert result.oral == expected_oral


async def test_get_molecule_raises_on_data_source_error():
    client = ChEMBLClient()
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(side_effect=DataSourceError("chembl", "HTTP 404", 404)),
    ):
        with pytest.raises(DataSourceError):
            await client.get_molecule("CHEMBL999999")


async def test_get_molecule_null_atc_returns_empty_list():
    fixture = {**CHEMBL894_FIXTURE, "atc_classifications": None}
    client = ChEMBLClient()
    with patch.object(client, "_rest_get", new=AsyncMock(return_value=fixture)):
        result = await client.get_molecule("CHEMBL894")

    assert result.atc_classifications == []
