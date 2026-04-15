"""Unit tests for services/approval_check — no network, no LLM calls."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.services.approval_check import (
    extract_approved_from_labels,
    get_fda_approved_diseases,
)


# --- extract_approved_from_labels ---


async def test_extract_approved_returns_matched_candidates(tmp_path):
    llm_response = json.dumps(["obesity", "type 2 diabetes mellitus"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity and type 2 diabetes."],
            candidate_diseases=["obesity", "type 2 diabetes mellitus", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity", "type 2 diabetes mellitus"}


async def test_extract_approved_returns_empty_on_no_matches(tmp_path):
    llm_response = json.dumps([])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for condition X."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == set()


async def test_extract_approved_empty_label_texts_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        result = await extract_approved_from_labels(
            label_texts=[],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert result == set()
    mock_llm.assert_not_awaited()


async def test_extract_approved_empty_candidates_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        result = await extract_approved_from_labels(
            label_texts=["Some label text."],
            candidate_diseases=[],
            cache_dir=tmp_path,
        )

    assert result == set()
    mock_llm.assert_not_awaited()


async def test_extract_approved_strips_markdown_fences(tmp_path):
    llm_response = '```json\n["obesity"]\n```'

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_rejects_hallucinated_disease(tmp_path):
    llm_response = json.dumps(["obesity", "made up disease"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_invalid_json_returns_empty(tmp_path):
    llm_response = "This is not valid JSON at all"

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert result == set()


async def test_extract_approved_case_insensitive_matching(tmp_path):
    llm_response = json.dumps(["Obesity"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_caches_result(tmp_path):
    llm_response = json.dumps(["obesity"])
    mock_llm = AsyncMock(return_value=llm_response)

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        first = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )
        second = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert first == {"obesity"}
    assert second == {"obesity"}
    mock_llm.assert_awaited_once()


# --- get_fda_approved_diseases ---


async def test_get_fda_approved_diseases_end_to_end(tmp_path):
    mock_fda_client = AsyncMock()
    mock_fda_client.__aenter__ = AsyncMock(return_value=mock_fda_client)
    mock_fda_client.__aexit__ = AsyncMock(return_value=None)
    mock_fda_client.get_all_label_indications = AsyncMock(
        return_value=["Indicated for obesity."]
    )

    with (
        patch(
            "indication_scout.services.approval_check.FDAClient",
            return_value=mock_fda_client,
        ),
        patch(
            "indication_scout.services.approval_check.extract_approved_from_labels",
            new=AsyncMock(return_value={"obesity"}),
        ) as mock_extract,
    ):
        result = await get_fda_approved_diseases(
            trade_names=["wegovy"],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_get_fda_approved_diseases_empty_trade_names(tmp_path):
    result = await get_fda_approved_diseases(
        trade_names=[],
        candidate_diseases=["obesity"],
        cache_dir=tmp_path,
    )
    assert result == set()


async def test_get_fda_approved_diseases_empty_candidates(tmp_path):
    result = await get_fda_approved_diseases(
        trade_names=["wegovy"],
        candidate_diseases=[],
        cache_dir=tmp_path,
    )
    assert result == set()


async def test_get_fda_approved_diseases_no_labels_found(tmp_path):
    mock_fda_client = AsyncMock()
    mock_fda_client.__aenter__ = AsyncMock(return_value=mock_fda_client)
    mock_fda_client.__aexit__ = AsyncMock(return_value=None)
    mock_fda_client.get_all_label_indications = AsyncMock(return_value=[])

    with patch(
        "indication_scout.services.approval_check.FDAClient",
        return_value=mock_fda_client,
    ):
        result = await get_fda_approved_diseases(
            trade_names=["unknownbrand"],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert result == set()
