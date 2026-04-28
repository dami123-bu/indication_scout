"""Integration tests for approval_check service — hits real openFDA + LLM APIs."""

import pytest

from indication_scout.services.approval_check import (
    get_all_fda_approved_diseases,
    get_fda_approved_disease_mapping,
    remove_approved_from_labels,
)


async def test_semaglutide_NASH(test_cache_dir):

    result = await get_all_fda_approved_diseases(
        drug_names=["semaglutide"],
        cache_dir=test_cache_dir,
    )

    assert result

# TODO delete
async def test_semaglutide(test_cache_dir):

    result = await get_all_fda_approved_diseases(
        drug_names=["semaglutide"],
        cache_dir=test_cache_dir,
    )

    assert result


@pytest.mark.parametrize(
    "drug_names, candidates, expected_survivors",
    [
        # Morbid-obesity regression: Wegovy is approved for obesity; morbid
        # obesity is a clinically-contained narrower subset — both should be
        # removed. Alzheimer's and hypertension are unrelated and survive.
        (
            ["wegovy", "semaglutide"],
            ["obesity", "morbid obesity", "alzheimer's disease", "hypertension"],
            {"alzheimer's disease", "hypertension"},
        ),
        # NASH synonym case: NASH = MASH. Probes whether the label (if any)
        # for semaglutide/Wegovy covers MASH and whether NASH is treated as a
        # synonym and removed.
        (
            ["wegovy", "semaglutide"],
            ["NASH", "MASH"],
            set(),
        ),
        # NAFLD parent-of-MASH case: NAFLD is a broader parent of MASH; even
        # if MASH is approved, NAFLD must survive (broader parent).
        (
            ["wegovy", "semaglutide"],
            ["non-alcoholic fatty liver disease","NAFLD"],
            {"non-alcoholic fatty liver disease"},
        ),
        # Parent-of-approval case: Ozempic is approved for type 2 diabetes
        # mellitus; "diabetes mellitus" is a broader parent that includes
        # T1DM and must survive.
        (
            ["ozempic"],
            ["type 2 diabetes mellitus", "diabetes mellitus"],
            {"diabetes mellitus"},
        ),
    ],
)
async def test_remove_approved_from_labels(
    test_cache_dir, drug_names, candidates, expected_survivors
):
    """Verify the function returns live repurposing candidates (approved ones removed)."""
    result = await remove_approved_from_labels(
        drug_names=drug_names,
        candidate_diseases=candidates,
        cache_dir=test_cache_dir,
    )
    assert result == expected_survivors


@pytest.mark.parametrize(
    "drug_name, candidates, expected",
    [
        # Curated short-circuit (approved): "morbid obesity" is in
        # CURATED_FDA_APPROVED_CANDIDATES["semaglutide"] → forced True without
        # an LLM call. "alzheimer's disease" falls through to the LLM path
        # and should come back False (semaglutide is not approved for AD).
        (
            "semaglutide",
            ["morbid obesity", "alzheimer's disease"],
            {"morbid obesity": True, "alzheimer's disease": False},
        ),
        # Curated short-circuit (rejected): "ischemic stroke" is in
        # CURATED_FDA_REJECTED_CANDIDATES["semaglutide"] → forced False.
        # "type 2 diabetes mellitus" goes through the LLM path against the
        # real Ozempic/Rybelsus labels and should come back True.
        (
            "semaglutide",
            ["ischemic stroke", "type 2 diabetes mellitus"],
            {"ischemic stroke": False, "type 2 diabetes mellitus": True},
        ),
        # Pure LLM path: trade-name input expanded via ChEMBL aliases.
        # Ozempic is FDA-approved for T2DM; hypertension is unrelated.
        (
            "ozempic",
            ["type 2 diabetes mellitus", "hypertension"],
            {"type 2 diabetes mellitus": True, "hypertension": False},
        ),
    ],
)
async def test_get_fda_approved_disease_mapping(
    test_cache_dir, drug_name, candidates, expected
):
    """Verify per-candidate FDA-approval mapping across curated and LLM paths."""
    result = await get_fda_approved_disease_mapping(
        drug_name=drug_name,
        candidate_diseases=candidates,
        cache_dir=test_cache_dir,
    )
    assert result == expected


@pytest.mark.parametrize(
    "drug_name, candidates, expected",
    [
        # Empty drug_name → early return; every candidate keyed False.
        ("", ["obesity"], {"obesity": False}),
        # Empty candidates → early return; empty dict.
        ("semaglutide", [], {}),
    ],
)
async def test_get_fda_approved_disease_mapping_empty_inputs(
    test_cache_dir, drug_name, candidates, expected
):
    """Empty drug_name or empty candidate list short-circuits without API calls."""
    result = await get_fda_approved_disease_mapping(
        drug_name=drug_name,
        candidate_diseases=candidates,
        cache_dir=test_cache_dir,
    )
    assert result == expected
