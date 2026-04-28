"""Integration tests for approval_check service — hits real openFDA + LLM APIs."""

import pytest

from indication_scout.data_sources.chembl import get_all_drug_names
from indication_scout.data_sources.fda import FDAClient
from indication_scout.services.approval_check import (
    get_all_fda_approved_diseases,
    get_fda_approved_disease_mapping,
    list_approved_indications_from_labels,
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


@pytest.mark.parametrize(
    "drug_aliases, must_contain_substrings",
    [
        # Semaglutide aliases (Ozempic = T2DM; Wegovy = chronic weight management
        # / obesity + cardiovascular risk reduction; Wegovy 2024 expansion = MASH).
        # We do not pin exact strings because the LLM controls phrasing — instead
        # we assert that each well-known approved indication has at least one
        # substring match in the returned list.
        (
            ["Ozempic", "Wegovy", "semaglutide"],
            [
                ["diabetes"],                            # T2DM approval
                ["weight", "obesity"],                   # chronic weight management / obesity
                ["cardiovascular", "cv"],                # 2020 MACE risk reduction approval
                ["kidney", "renal"],                     # 2025 FLOW / CKD approval
                ["mash", "steatohepatitis", "fatty liver"],  # 2024 Wegovy MASH approval
            ],
        ),
        # Metformin: established T2DM approval. Single, unambiguous indication
        # makes this a low-noise sanity check that the path runs end-to-end.
        (
            ["metformin"],
            [["diabetes"]],
        ),
    ],
)
async def test_list_approved_indications_from_labels_real_drug(
    test_cache_dir, drug_aliases, must_contain_substrings
):
    """End-to-end: fetch real openFDA labels and extract approved indications via the LLM.

    Asserts that each expected approval appears (case-insensitively) somewhere
    in the returned list — at least one of the substrings in each group must
    match at least one returned indication.
    """
    async with FDAClient(cache_dir=test_cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_aliases)

    assert label_texts, f"openFDA returned no label text for {drug_aliases}"

    indications = await list_approved_indications_from_labels(
        label_texts=label_texts,
        cache_dir=test_cache_dir,
    )

    assert isinstance(indications, list)
    assert all(isinstance(x, str) and x for x in indications), indications
    assert len(indications) >= 1, f"no indications extracted: {indications}"

    lowered = [i.lower() for i in indications]
    for substring_group in must_contain_substrings:
        assert any(
            any(sub in ind for sub in substring_group) for ind in lowered
        ), (
            f"none of {substring_group} found in extracted indications "
            f"{indications} for {drug_aliases}"
        )


async def test_list_approved_indications_from_labels_manual(test_cache_dir):
    """Manual hand-test slot — edit drug_aliases / expectations in place.

    Same shape as test_list_approved_indications_from_labels_real_drug but
    unparametrized so you can tweak the drug, the expected substring groups,
    and the cache path freely without disturbing the regression cases.
    """
    # drug_aliases = ["dasatinib"]
    drug_aliases = await get_all_drug_names("CHEMBL941", test_cache_dir)
    # must_contain_substrings = [
    #     ["diabetes"],
    #     ["weight", "obesity"],
    # ]

    async with FDAClient(cache_dir=test_cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_aliases)

    assert label_texts, f"openFDA returned no label text for {drug_aliases}"

    indications = await list_approved_indications_from_labels(
        label_texts=label_texts,
        cache_dir=test_cache_dir,
    )
    assert indications

    # assert isinstance(indications, list)
    # assert all(isinstance(x, str) and x for x in indications), indications
    # assert len(indications) >= 1, f"no indications extracted: {indications}"
    #
    # lowered = [i.lower() for i in indications]
    # for substring_group in must_contain_substrings:
    #     assert any(
    #         any(sub in ind for sub in substring_group) for ind in lowered
    #     ), (
    #         f"none of {substring_group} found in extracted indications "
    #         f"{indications} for {drug_aliases}"
    #     )


async def test_list_approved_indications_from_labels_empty_input(test_cache_dir):
    """Empty label_texts short-circuits without an LLM call and returns []."""
    result = await list_approved_indications_from_labels(
        label_texts=[],
        cache_dir=test_cache_dir,
    )
    assert result == []
