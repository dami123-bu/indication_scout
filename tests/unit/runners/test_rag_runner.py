"""Unit tests for runners/rag_runner — no network, no LLM, no DB calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.runners.rag_runner import run_rag

# --- Fixtures ---


@pytest.fixture
def mock_db() -> MagicMock:
    return MagicMock()


@pytest.fixture
def drug_profile() -> DrugProfile:
    return DrugProfile(
        name="metformin",
        synonyms=["Glucophage"],
        target_gene_symbols=["PRKAA1"],
        mechanisms_of_action=["AMP-activated protein kinase activator"],
        atc_codes=["A10BA02"],
        atc_descriptions=["Biguanides"],
        drug_type="Small molecule",
    )


@pytest.fixture
def evidence_colorectal() -> EvidenceSummary:
    return EvidenceSummary(
        summary="Evidence for metformin in colorectal cancer.",
        study_count=3,
        study_types=["RCT"],
        strength="moderate",
        key_findings=["Reduced CRC incidence"],
        supporting_pmids=["11111111"],
    )


@pytest.fixture
def evidence_breast() -> EvidenceSummary:
    return EvidenceSummary(
        summary="Evidence for metformin in breast cancer.",
        study_count=2,
        study_types=["observational"],
        strength="weak",
        key_findings=["Lower recurrence rate"],
        supporting_pmids=["22222222"],
    )


# --- Tests ---


async def test_run_rag_returns_evidence_per_disease(
    mock_db, drug_profile, evidence_colorectal, evidence_breast
):
    """run_rag returns a dict mapping each disease to its EvidenceSummary."""
    top_10 = {"colorectal cancer": {"aspirin"}, "breast cancer": {"tamoxifen"}}

    mock_ot = AsyncMock()
    mock_ot.__aenter__ = AsyncMock(return_value=mock_ot)
    mock_ot.__aexit__ = AsyncMock(return_value=None)
    mock_ot.get_drug_competitors = AsyncMock(return_value=top_10)

    with (
        patch(
            "indication_scout.runners.rag_runner.OpenTargetsClient",
            return_value=mock_ot,
        ),
        patch(
            "indication_scout.runners.rag_runner.build_drug_profile",
            new=AsyncMock(return_value=drug_profile),
        ),
        patch(
            "indication_scout.runners.rag_runner.expand_search_terms",
            new=AsyncMock(return_value=["metformin AND colorectal cancer"]),
        ),
        patch(
            "indication_scout.runners.rag_runner.fetch_and_cache",
            new=AsyncMock(return_value=["11111111", "22222222"]),
        ),
        patch(
            "indication_scout.runners.rag_runner.semantic_search",
            new=AsyncMock(
                side_effect=[
                    [
                        {
                            "pmid": "11111111",
                            "title": "T",
                            "abstract": "A",
                            "similarity": 0.9,
                        }
                    ],
                    [
                        {
                            "pmid": "22222222",
                            "title": "T2",
                            "abstract": "A2",
                            "similarity": 0.8,
                        }
                    ],
                ]
            ),
        ),
        patch(
            "indication_scout.runners.rag_runner.synthesize",
            new=AsyncMock(side_effect=[evidence_colorectal, evidence_breast]),
        ),
    ):
        results = await run_rag("metformin", mock_db)

    assert set(results.keys()) == {"colorectal cancer", "breast cancer"}
    assert results["colorectal cancer"] is evidence_colorectal
    assert results["breast cancer"] is evidence_breast


async def test_run_rag_passes_drug_profile_to_expand_search_terms(
    mock_db, drug_profile
):
    """expand_search_terms receives the DrugProfile returned by build_drug_profile."""
    top_10 = {"colorectal cancer": {"aspirin"}}

    mock_ot = AsyncMock()
    mock_ot.__aenter__ = AsyncMock(return_value=mock_ot)
    mock_ot.__aexit__ = AsyncMock(return_value=None)
    mock_ot.get_drug_competitors = AsyncMock(return_value=top_10)

    mock_expand = AsyncMock(return_value=["q"])

    with (
        patch(
            "indication_scout.runners.rag_runner.OpenTargetsClient",
            return_value=mock_ot,
        ),
        patch(
            "indication_scout.runners.rag_runner.build_drug_profile",
            new=AsyncMock(return_value=drug_profile),
        ),
        patch(
            "indication_scout.runners.rag_runner.expand_search_terms", new=mock_expand
        ),
        patch(
            "indication_scout.runners.rag_runner.fetch_and_cache",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "indication_scout.runners.rag_runner.semantic_search",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "indication_scout.runners.rag_runner.synthesize",
            new=AsyncMock(return_value=EvidenceSummary()),
        ),
    ):
        await run_rag("metformin", mock_db)

    mock_expand.assert_called_once_with("metformin", "colorectal cancer", drug_profile)


async def test_run_rag_passes_pmids_to_semantic_search(mock_db, drug_profile):
    """semantic_search receives the PMID list returned by fetch_and_cache."""
    top_10 = {"colorectal cancer": {"aspirin"}}
    pmids = ["11111111", "22222222", "33333333"]

    mock_ot = AsyncMock()
    mock_ot.__aenter__ = AsyncMock(return_value=mock_ot)
    mock_ot.__aexit__ = AsyncMock(return_value=None)
    mock_ot.get_drug_competitors = AsyncMock(return_value=top_10)

    mock_search = AsyncMock(return_value=[])

    with (
        patch(
            "indication_scout.runners.rag_runner.OpenTargetsClient",
            return_value=mock_ot,
        ),
        patch(
            "indication_scout.runners.rag_runner.build_drug_profile",
            new=AsyncMock(return_value=drug_profile),
        ),
        patch(
            "indication_scout.runners.rag_runner.expand_search_terms",
            new=AsyncMock(return_value=["q"]),
        ),
        patch(
            "indication_scout.runners.rag_runner.fetch_and_cache",
            new=AsyncMock(return_value=pmids),
        ),
        patch("indication_scout.runners.rag_runner.semantic_search", new=mock_search),
        patch(
            "indication_scout.runners.rag_runner.synthesize",
            new=AsyncMock(return_value=EvidenceSummary()),
        ),
    ):
        await run_rag("metformin", mock_db)

    mock_search.assert_called_once_with(
        "colorectal cancer", "metformin", pmids, mock_db
    )


async def test_run_rag_passes_top_abstracts_to_synthesize(mock_db, drug_profile):
    """synthesize receives the abstract list returned by semantic_search."""
    top_10 = {"colorectal cancer": {"aspirin"}}
    top_abstracts = [
        {"pmid": "11111111", "title": "T", "abstract": "A", "similarity": 0.9}
    ]

    mock_ot = AsyncMock()
    mock_ot.__aenter__ = AsyncMock(return_value=mock_ot)
    mock_ot.__aexit__ = AsyncMock(return_value=None)
    mock_ot.get_drug_competitors = AsyncMock(return_value=top_10)

    mock_synthesize = AsyncMock(return_value=EvidenceSummary())

    with (
        patch(
            "indication_scout.runners.rag_runner.OpenTargetsClient",
            return_value=mock_ot,
        ),
        patch(
            "indication_scout.runners.rag_runner.build_drug_profile",
            new=AsyncMock(return_value=drug_profile),
        ),
        patch(
            "indication_scout.runners.rag_runner.expand_search_terms",
            new=AsyncMock(return_value=["q"]),
        ),
        patch(
            "indication_scout.runners.rag_runner.fetch_and_cache",
            new=AsyncMock(return_value=["11111111"]),
        ),
        patch(
            "indication_scout.runners.rag_runner.semantic_search",
            new=AsyncMock(return_value=top_abstracts),
        ),
        patch("indication_scout.runners.rag_runner.synthesize", new=mock_synthesize),
    ):
        await run_rag("metformin", mock_db)

    mock_synthesize.assert_called_once_with(
        "metformin", "colorectal cancer", top_abstracts
    )
