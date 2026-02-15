"""Unit tests for OpenTargetsClient."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import (
    DrugData,
    DrugSummary,
    DrugTarget,
)


# --- OpenTargetsClient configuration ---


def test_default_config():
    """Test that client uses default settings."""
    client = OpenTargetsClient()

    assert client.timeout == 30.0
    assert client.max_retries == 3
    assert client.cache_dir is not None
    assert client.cache_dir == Path("_cache")


# --- get_drug_target_competitors ---


@pytest.mark.asyncio
async def test_get_drug_target_competitors_multiple_targets():
    """get_drug_target_competitors should return dict mapping target symbols to drug summaries."""
    client = OpenTargetsClient()

    mock_drug = DrugData(
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
    )

    glp1r_summaries = [
        DrugSummary(
            drug_id="CHEMBL4084119",
            drug_name="LIRAGLUTIDE",
            disease_id="MONDO_0005148",
            disease_name="type 2 diabetes mellitus",
            phase=4.0,
            mechanism_of_action="Glucagon-like peptide 1 receptor agonist",
        ),
        DrugSummary(
            drug_id="CHEMBL2108724",
            drug_name="SEMAGLUTIDE",
            disease_id="MONDO_0005148",
            disease_name="type 2 diabetes mellitus",
            phase=4.0,
            mechanism_of_action="Glucagon-like peptide 1 receptor agonist",
        ),
    ]

    pygo1_summaries = [
        DrugSummary(
            drug_id="CHEMBL9999",
            drug_name="DRUG_X",
            disease_id="EFO_0001073",
            disease_name="obesity",
            phase=2.0,
            mechanism_of_action="Pygo homolog 1 inhibitor",
        ),
    ]

    with patch.object(
        client, "get_drug", new_callable=AsyncMock, return_value=mock_drug
    ), patch.object(
        client,
        "get_target_data_drug_summaries",
        new_callable=AsyncMock,
        side_effect=[glp1r_summaries, pygo1_summaries],
    ):
        result = await client.get_drug_target_competitors("semaglutide")

    assert set(result.keys()) == {"GLP1R", "PYGO1"}
    assert len(result["GLP1R"]) == 2
    assert result["GLP1R"][0].drug_name == "LIRAGLUTIDE"
    assert result["GLP1R"][1].drug_name == "SEMAGLUTIDE"
    assert len(result["PYGO1"]) == 1
    assert result["PYGO1"][0].drug_name == "DRUG_X"


@pytest.mark.asyncio
async def test_get_drug_target_competitors_no_targets():
    """get_drug_target_competitors should return empty dict when drug has no targets."""
    client = OpenTargetsClient()

    mock_drug = DrugData(
        chembl_id="CHEMBL9999",
        name="NO_TARGET_DRUG",
        drug_type="Small molecule",
        is_approved=False,
        max_clinical_phase=1.0,
        targets=[],
    )

    with patch.object(
        client, "get_drug", new_callable=AsyncMock, return_value=mock_drug
    ):
        result = await client.get_drug_target_competitors("no_target_drug")

    assert result == {}


@pytest.mark.asyncio
async def test_get_drug_target_competitors_single_target_empty_summaries():
    """get_drug_target_competitors should handle targets with no known drugs."""
    client = OpenTargetsClient()

    mock_drug = DrugData(
        chembl_id="CHEMBL1234",
        name="TEST_DRUG",
        drug_type="Small molecule",
        is_approved=False,
        max_clinical_phase=1.0,
        targets=[
            DrugTarget(
                target_id="ENSG00000000001",
                target_symbol="GENE1",
                mechanism_of_action="Some mechanism",
            ),
        ],
    )

    with patch.object(
        client, "get_drug", new_callable=AsyncMock, return_value=mock_drug
    ), patch.object(
        client,
        "get_target_data_drug_summaries",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await client.get_drug_target_competitors("test_drug")

    assert result == {"GENE1": []}
