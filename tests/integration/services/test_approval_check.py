"""Integration tests for approval_check service — hits real openFDA + LLM APIs."""

import pytest

from indication_scout.services.approval_check import (
    get_all_fda_approved_diseases,
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
