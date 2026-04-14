"""Unit tests for ChEMBLClient.get_molecule(), get_atc_description(), and get_trade_names()."""

from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.models.model_chembl import ATCDescription, MoleculeData, MoleculeSynonym

CHEMBL894_FIXTURE = {
    "molecule_chembl_id": "CHEMBL894",
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": ["N06AX12"],
    "black_box_warning": 1,
    "first_approval": 1985,
    "oral": True,
    "molecule_synonyms": [
        {"molecule_synonym": "Bupropion", "syn_type": "INN", "synonyms": "BUPROPION"},
    ],
}

CHEMBL2108724_FIXTURE = {
    "molecule_chembl_id": "CHEMBL2108724",
    "molecule_type": "Protein",
    "max_phase": "4.0",
    "atc_classifications": ["A10BJ06"],
    "black_box_warning": 1,
    "first_approval": 2017,
    "oral": True,
    "molecule_synonyms": [
        {"molecule_synonym": "Ozempic", "syn_type": "TRADE_NAME", "synonyms": "OZEMPIC"},
        {"molecule_synonym": "Semaglutide", "syn_type": "INN", "synonyms": "SEMAGLUTIDE"},
    ],
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
    "molecule_synonyms": [],
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
    "molecule_synonyms": [],
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
    "molecule_synonyms": None,
}


@pytest.mark.parametrize(
    "chembl_id, fixture, expected_atc, expected_type, expected_max_phase, expected_black_box, expected_first_approval, expected_oral",
    [
        (
            "CHEMBL894",
            CHEMBL894_FIXTURE,
            ["N06AX12"],
            "Small molecule",
            "4.0",
            1,
            1985,
            True,
        ),
        (
            "CHEMBL2108724",
            CHEMBL2108724_FIXTURE,
            ["A10BJ06"],
            "Protein",
            "4.0",
            1,
            2017,
            True,
        ),
        (
            "CHEMBL192",
            CHEMBL192_FIXTURE,
            ["G04BE03"],
            "Small molecule",
            "4.0",
            0,
            1998,
            True,
        ),
        (
            "CHEMBL1201585",
            CHEMBL1201585_FIXTURE,
            ["L01FD01"],
            "Antibody",
            "4.0",
            1,
            1998,
            False,
        ),
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


# --- get_atc_description ---

ATC_A10BA02_FIXTURE = {
    "level1": "A",
    "level1_description": "ALIMENTARY TRACT AND METABOLISM",
    "level2": "A10",
    "level2_description": "DRUGS USED IN DIABETES",
    "level3": "A10B",
    "level3_description": "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
    "level4": "A10BA",
    "level4_description": "Biguanides",
    "level5": "A10BA02",
    "who_name": "metformin",
}

ATC_N06AX12_FIXTURE = {
    "level1": "N",
    "level1_description": "NERVOUS SYSTEM",
    "level2": "N06",
    "level2_description": "PSYCHOANALEPTICS",
    "level3": "N06A",
    "level3_description": "ANTIDEPRESSANTS",
    "level4": "N06AX",
    "level4_description": "Other antidepressants",
    "level5": "N06AX12",
    "who_name": "bupropion",
}


@pytest.mark.parametrize(
    "atc_code, fixture, expected_level1, expected_level1_description, expected_level2, expected_level2_description, expected_level3, expected_level3_description, expected_level4, expected_level4_description, expected_level5, expected_who_name",
    [
        (
            "A10BA02",
            ATC_A10BA02_FIXTURE,
            "A",
            "ALIMENTARY TRACT AND METABOLISM",
            "A10",
            "DRUGS USED IN DIABETES",
            "A10B",
            "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
            "A10BA",
            "Biguanides",
            "A10BA02",
            "metformin",
        ),
        (
            "N06AX12",
            ATC_N06AX12_FIXTURE,
            "N",
            "NERVOUS SYSTEM",
            "N06",
            "PSYCHOANALEPTICS",
            "N06A",
            "ANTIDEPRESSANTS",
            "N06AX",
            "Other antidepressants",
            "N06AX12",
            "bupropion",
        ),
    ],
)
async def test_get_atc_description(
    atc_code,
    fixture,
    expected_level1,
    expected_level1_description,
    expected_level2,
    expected_level2_description,
    expected_level3,
    expected_level3_description,
    expected_level4,
    expected_level4_description,
    expected_level5,
    expected_who_name,
):
    client = ChEMBLClient()
    with patch.object(client, "_rest_get", new=AsyncMock(return_value=fixture)):
        result = await client.get_atc_description(atc_code)

    assert isinstance(result, ATCDescription)
    assert result.level1 == expected_level1
    assert result.level1_description == expected_level1_description
    assert result.level2 == expected_level2
    assert result.level2_description == expected_level2_description
    assert result.level3 == expected_level3
    assert result.level3_description == expected_level3_description
    assert result.level4 == expected_level4
    assert result.level4_description == expected_level4_description
    assert result.level5 == expected_level5
    assert result.who_name == expected_who_name


async def test_get_atc_description_raises_on_data_source_error(tmp_path):
    client = ChEMBLClient(cache_dir=tmp_path)
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(side_effect=DataSourceError("chembl", "HTTP 404", 404)),
    ):
        with pytest.raises(DataSourceError):
            await client.get_atc_description("A10BA02")


# --- get_trade_names ---

# Parent molecule (bupropion freebase) — no TRADE_NAME synonyms on parent
PARENT_FIXTURE = {
    "molecule_chembl_id": "CHEMBL894",
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": ["N06AX12"],
    "black_box_warning": 1,
    "first_approval": 1985,
    "oral": True,
    "molecule_synonyms": [
        {"molecule_synonym": "Bupropion", "syn_type": "INN", "synonyms": "BUPROPION"},
        {"molecule_synonym": "BW-323", "syn_type": "RESEARCH_CODE", "synonyms": "BW-323"},
    ],
}

# Salt form (bupropion HCl) — has TRADE_NAME synonyms
SALT_HCL_MOLECULE = {
    "molecule_chembl_id": "CHEMBL1698",
    "molecule_synonyms": [
        {"molecule_synonym": "Wellbutrin", "syn_type": "TRADE_NAME", "synonyms": "WELLBUTRIN"},
        {"molecule_synonym": "Zyban", "syn_type": "TRADE_NAME", "synonyms": "ZYBAN"},
        {"molecule_synonym": "Bupropion hydrochloride", "syn_type": "INN", "synonyms": "BUPROPION HYDROCHLORIDE"},
        {"molecule_synonym": "Bupropion hydrochloride component of contrave", "syn_type": "TRADE_NAME", "synonyms": "BUPROPION HYDROCHLORIDE COMPONENT OF CONTRAVE"},
    ],
}

SALT_HBR_MOLECULE = {
    "molecule_chembl_id": "CHEMBL1201735",
    "molecule_synonyms": [
        {"molecule_synonym": "Aplenzin", "syn_type": "TRADE_NAME", "synonyms": "APLENZIN"},
    ],
}

HIERARCHY_RESPONSE = {
    "molecules": [
        {"molecule_chembl_id": "CHEMBL894", "molecule_synonyms": PARENT_FIXTURE["molecule_synonyms"]},
        SALT_HCL_MOLECULE,
        SALT_HBR_MOLECULE,
    ],
}

# Biologic (semaglutide) — trade names directly on parent, no salts
BIOLOGIC_PARENT_FIXTURE = {
    "molecule_chembl_id": "CHEMBL2108724",
    "molecule_type": "Protein",
    "max_phase": "4.0",
    "atc_classifications": ["A10BJ06"],
    "black_box_warning": 1,
    "first_approval": 2017,
    "oral": True,
    "molecule_synonyms": [
        {"molecule_synonym": "Ozempic", "syn_type": "TRADE_NAME", "synonyms": "OZEMPIC"},
        {"molecule_synonym": "Rybelsus", "syn_type": "TRADE_NAME", "synonyms": "RYBELSUS"},
        {"molecule_synonym": "Wegovy", "syn_type": "TRADE_NAME", "synonyms": "WEGOVY"},
        {"molecule_synonym": "Semaglutide", "syn_type": "INN", "synonyms": "SEMAGLUTIDE"},
    ],
}

BIOLOGIC_HIERARCHY_RESPONSE = {
    "molecules": [
        {"molecule_chembl_id": "CHEMBL2108724", "molecule_synonyms": BIOLOGIC_PARENT_FIXTURE["molecule_synonyms"]},
    ],
}


async def test_get_trade_names_small_molecule_with_salts(tmp_path):
    """Trade names come from salt forms, not the parent freebase molecule."""
    client = ChEMBLClient(cache_dir=tmp_path)

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL894.json" in url:
            return PARENT_FIXTURE
        if "/molecule.json" in url:
            return HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with patch.object(client, "_rest_get", new=AsyncMock(side_effect=mock_rest_get)):
        result = await client.get_trade_names("CHEMBL894")

    assert result == ["Wellbutrin", "Zyban", "Aplenzin"]


async def test_get_trade_names_biologic_no_salts(tmp_path):
    """Biologics have trade names on the parent molecule directly."""
    client = ChEMBLClient(cache_dir=tmp_path)

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return BIOLOGIC_PARENT_FIXTURE
        if "/molecule.json" in url:
            return BIOLOGIC_HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with patch.object(client, "_rest_get", new=AsyncMock(side_effect=mock_rest_get)):
        result = await client.get_trade_names("CHEMBL2108724")

    assert result == ["Ozempic", "Rybelsus", "Wegovy"]


async def test_get_trade_names_filters_component_of(tmp_path):
    """Entries containing 'component of' are excluded."""
    client = ChEMBLClient(cache_dir=tmp_path)

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL894.json" in url:
            return PARENT_FIXTURE
        if "/molecule.json" in url:
            return HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with patch.object(client, "_rest_get", new=AsyncMock(side_effect=mock_rest_get)):
        result = await client.get_trade_names("CHEMBL894")

    assert "Bupropion hydrochloride component of contrave" not in result


async def test_get_trade_names_no_synonyms_returns_empty(tmp_path):
    """Molecule with no synonyms returns empty list."""
    client = ChEMBLClient(cache_dir=tmp_path)
    bare_fixture = {
        "molecule_chembl_id": "CHEMBL999",
        "molecule_type": "Small molecule",
        "max_phase": None,
        "atc_classifications": [],
        "black_box_warning": 0,
        "first_approval": None,
        "oral": False,
        "molecule_synonyms": [],
    }

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL999.json" in url:
            return bare_fixture
        if "/molecule.json" in url:
            return {"molecules": []}
        raise AssertionError(f"Unexpected URL: {url}")

    with patch.object(client, "_rest_get", new=AsyncMock(side_effect=mock_rest_get)):
        result = await client.get_trade_names("CHEMBL999")

    assert result == []


async def test_get_trade_names_hierarchy_failure_still_returns_parent_names(tmp_path):
    """If hierarchy lookup fails, still returns trade names from the parent."""
    client = ChEMBLClient(cache_dir=tmp_path)

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return BIOLOGIC_PARENT_FIXTURE
        if "/molecule.json" in url:
            raise DataSourceError("chembl", "HTTP 500", 500)
        raise AssertionError(f"Unexpected URL: {url}")

    with patch.object(client, "_rest_get", new=AsyncMock(side_effect=mock_rest_get)):
        result = await client.get_trade_names("CHEMBL2108724")

    assert result == ["Ozempic", "Rybelsus", "Wegovy"]


async def test_get_trade_names_deduplicates(tmp_path):
    """Same trade name from parent and salt is not repeated."""
    client = ChEMBLClient(cache_dir=tmp_path)
    parent = {
        **BIOLOGIC_PARENT_FIXTURE,
        "molecule_synonyms": [
            {"molecule_synonym": "Ozempic", "syn_type": "TRADE_NAME", "synonyms": "OZEMPIC"},
        ],
    }
    hierarchy = {
        "molecules": [
            {
                "molecule_chembl_id": "CHEMBL_SALT",
                "molecule_synonyms": [
                    {"molecule_synonym": "Ozempic", "syn_type": "TRADE_NAME", "synonyms": "OZEMPIC"},
                ],
            },
        ],
    }

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return parent
        if "/molecule.json" in url:
            return hierarchy
        raise AssertionError(f"Unexpected URL: {url}")

    with patch.object(client, "_rest_get", new=AsyncMock(side_effect=mock_rest_get)):
        result = await client.get_trade_names("CHEMBL2108724")

    assert result == ["Ozempic"]


async def test_get_trade_names_caches_result(tmp_path):
    """Second call uses cache, no additional API calls."""
    client = ChEMBLClient(cache_dir=tmp_path)

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return BIOLOGIC_PARENT_FIXTURE
        if "/molecule.json" in url:
            return BIOLOGIC_HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    mock = AsyncMock(side_effect=mock_rest_get)
    with patch.object(client, "_rest_get", new=mock):
        first = await client.get_trade_names("CHEMBL2108724")
        second = await client.get_trade_names("CHEMBL2108724")

    assert first == second == ["Ozempic", "Rybelsus", "Wegovy"]
    # get_molecule calls _rest_get once, hierarchy calls once = 2 total
    assert mock.await_count == 2
