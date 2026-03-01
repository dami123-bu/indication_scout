"""Unit tests for services/retrieval — no network, no LLM calls."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.models.model_chembl import ATCDescription
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_open_targets import (
    DrugData,
    DrugTarget,
    RichDrugData,
    TargetData,
)
from indication_scout.services.retrieval import expand_search_terms, extract_organ_term

# --- Fixtures ---


@pytest.fixture
def atc_metformin() -> ATCDescription:
    return ATCDescription(
        level1="A",
        level1_description="ALIMENTARY TRACT AND METABOLISM",
        level2="A10",
        level2_description="DRUGS USED IN DIABETES",
        level3="A10B",
        level3_description="BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
        level4="A10BA",
        level4_description="Biguanides",
        level5="A10BA02",
        who_name="metformin",
    )


@pytest.fixture
def rich_metformin(atc_metformin) -> RichDrugData:
    drug = DrugData(
        chembl_id="CHEMBL1431",
        name="METFORMIN",
        synonyms=["Glucophage", "Glucophage"],  # intentional duplicate
        trade_names=["Fortamet", "Glucophage"],  # overlap with synonyms
        drug_type="Small molecule",
        is_approved=True,
        max_clinical_phase=4.0,
        atc_classifications=["A10BA02"],
        targets=[
            DrugTarget(
                target_id="ENSG00000132356",
                target_symbol="PRKAA1",
                mechanism_of_action="AMP-activated protein kinase activator",
                action_type="ACTIVATOR",
            ),
            DrugTarget(
                target_id="ENSG00000162409",
                target_symbol="PRKAA2",
                mechanism_of_action="AMP-activated protein kinase activator",  # duplicate MoA
                action_type="ACTIVATOR",
            ),
        ],
    )
    targets = [
        TargetData(
            target_id="ENSG00000132356",
            symbol="PRKAA1",
            name="Protein kinase AMP-activated alpha 1",
        ),
        TargetData(
            target_id="ENSG00000162409",
            symbol="PRKAA2",
            name="Protein kinase AMP-activated alpha 2",
        ),
    ]
    return RichDrugData(drug=drug, targets=targets)


@pytest.fixture
def metformin_profile() -> DrugProfile:
    return DrugProfile(
        name="metformin",
        synonyms=["Glucophage", "Fortamet"],
        target_gene_symbols=["PRKAA1", "PRKAA2", "STK11"],
        mechanisms_of_action=[
            "AMP-activated protein kinase activator",
            "mTOR inhibitor",
        ],
        atc_codes=["A10BA02"],
        atc_descriptions=["BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS", "Biguanides"],
        drug_type="Small molecule",
    )


# --- DrugProfile.from_rich_drug_data ---


def test_drug_profile_from_rich_drug_data_name(rich_metformin, atc_metformin):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.name == "METFORMIN"


def test_drug_profile_from_rich_drug_data_synonyms_deduped(
    rich_metformin, atc_metformin
):
    """synonyms = drug.synonyms + drug.trade_names, deduplicated, order-preserving."""
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    # synonyms: ["Glucophage", "Glucophage"] → deduped → ["Glucophage"]
    # trade_names: ["Fortamet", "Glucophage"] → "Glucophage" already seen
    assert profile.synonyms == ["Glucophage", "Fortamet"]


def test_drug_profile_from_rich_drug_data_target_gene_symbols(
    rich_metformin, atc_metformin
):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.target_gene_symbols == ["PRKAA1", "PRKAA2"]


def test_drug_profile_from_rich_drug_data_mechanisms_deduped(
    rich_metformin, atc_metformin
):
    """Duplicate MoA strings across targets are collapsed to one."""
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.mechanisms_of_action == ["AMP-activated protein kinase activator"]


def test_drug_profile_from_rich_drug_data_atc_codes(rich_metformin, atc_metformin):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.atc_codes == ["A10BA02"]


def test_drug_profile_from_rich_drug_data_atc_descriptions(
    rich_metformin, atc_metformin
):
    """level3_description then level4_description, deduplicated."""
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.atc_descriptions == [
        "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
        "Biguanides",
    ]


def test_drug_profile_from_rich_drug_data_drug_type(rich_metformin, atc_metformin):
    profile = DrugProfile.from_rich_drug_data(rich_metformin, [atc_metformin])
    assert profile.drug_type == "Small molecule"


# --- extract_organ_term ---


async def test_extract_organ_term_returns_stripped_string():
    with patch(
        "indication_scout.services.retrieval.query_small_llm",
        new=AsyncMock(return_value="  colon  "),
    ):
        result = await extract_organ_term("colorectal cancer")
    assert result == "colon"


async def test_extract_organ_term_returns_cached_result(tmp_path):
    from indication_scout.utils.cache import cache_set

    cache_set("organ_term", {"disease_name": "colorectal cancer"}, "colon", tmp_path)

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(),
        ) as mock_llm,
    ):
        result = await extract_organ_term("colorectal cancer")

    assert result == "colon"
    mock_llm.assert_not_called()


# --- expand_search_terms ---


async def test_expand_search_terms_returns_list(tmp_path, metformin_profile):
    llm_response = '["metformin AND colorectal cancer", "biguanides AND colon"]'
    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=llm_response),
        ),
    ):
        result = await expand_search_terms(
            "metformin", "colorectal cancer", metformin_profile
        )

    assert result == ["metformin AND colorectal cancer", "biguanides AND colon"]


async def test_expand_search_terms_prompt_contains_drug_name(
    tmp_path, metformin_profile
):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await expand_search_terms("metformin", "colorectal cancer", metformin_profile)

    assert "metformin" in captured["prompt"]
    assert "colorectal cancer" in captured["prompt"]


async def test_expand_search_terms_prompt_contains_targets(tmp_path, metformin_profile):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await expand_search_terms("metformin", "colorectal cancer", metformin_profile)

    assert "PRKAA1" in captured["prompt"]
    assert "PRKAA2" in captured["prompt"]
    assert "STK11" in captured["prompt"]


async def test_expand_search_terms_prompt_contains_atc_descriptions(
    tmp_path, metformin_profile
):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await expand_search_terms("metformin", "colorectal cancer", metformin_profile)

    assert "Biguanides" in captured["prompt"]
    assert "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS" in captured["prompt"]


async def test_expand_search_terms_prompt_contains_organ_term(
    tmp_path, metformin_profile
):
    captured = {}

    async def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '["metformin AND colorectal cancer"]'

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch("indication_scout.services.retrieval.query_small_llm", new=capture_llm),
    ):
        await expand_search_terms("metformin", "colorectal cancer", metformin_profile)

    assert "colon" in captured["prompt"]


async def test_expand_search_terms_deduplicates_output(tmp_path, metformin_profile):
    """Case-duplicate entries in LLM output are deduped; first occurrence casing is preserved."""
    llm_response = '["Metformin AND colorectal cancer", "metformin AND colorectal cancer", "biguanides AND colon"]'
    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.extract_organ_term",
            new=AsyncMock(return_value="colon"),
        ),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(return_value=llm_response),
        ),
    ):
        result = await expand_search_terms(
            "metformin", "colorectal cancer", metformin_profile
        )

    assert result == ["Metformin AND colorectal cancer", "biguanides AND colon"]


async def test_expand_search_terms_returns_cached_result(tmp_path, metformin_profile):
    from indication_scout.utils.cache import cache_set

    cached_queries = ["metformin AND colorectal cancer", "biguanides AND colon"]
    cache_set(
        "expand_search_terms",
        {"drug_name": "metformin", "disease_name": "colorectal cancer"},
        cached_queries,
        tmp_path,
    )

    with (
        patch("indication_scout.services.retrieval.DEFAULT_CACHE_DIR", tmp_path),
        patch(
            "indication_scout.services.retrieval.query_small_llm",
            new=AsyncMock(),
        ) as mock_llm,
    ):
        result = await expand_search_terms(
            "metformin", "colorectal cancer", metformin_profile
        )

    assert result == cached_queries
    mock_llm.assert_not_called()
