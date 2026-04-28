"""Unit tests for services/approval_check — no network, no LLM calls."""

import json
from unittest.mock import AsyncMock, patch

from indication_scout.services.approval_check import (
    extract_approved_from_labels,
    list_approved_indications_from_labels,
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


# --- list_approved_indications_from_labels ---


async def test_list_approved_indications_returns_parsed_list(tmp_path):
    llm_response = json.dumps(["type 2 diabetes mellitus", "chronic weight management"])

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=[
                "OZEMPIC is indicated as an adjunct to diet and exercise to "
                "improve glycemic control in adults with type 2 diabetes mellitus."
            ],
            cache_dir=tmp_path,
        )

    assert result == ["type 2 diabetes mellitus", "chronic weight management"]


async def test_list_approved_indications_empty_label_texts_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=mock_llm,
    ):
        result = await list_approved_indications_from_labels(
            label_texts=[],
            cache_dir=tmp_path,
        )

    assert result == []
    mock_llm.assert_not_awaited()


async def test_list_approved_indications_strips_markdown_fences(tmp_path):
    llm_response = '```json\n["MASH"]\n```'

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Indicated for MASH with moderate to advanced liver fibrosis."],
            cache_dir=tmp_path,
        )

    assert result == ["MASH"]


async def test_list_approved_indications_invalid_json_returns_empty(tmp_path):
    llm_response = "not valid json"

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label text."],
            cache_dir=tmp_path,
        )

    assert result == []


async def test_list_approved_indications_non_list_returns_empty(tmp_path):
    llm_response = json.dumps({"approved": ["obesity"]})

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )

    assert result == []


async def test_list_approved_indications_dedupes_case_insensitively(tmp_path):
    llm_response = json.dumps(["Obesity", "obesity", "  type 2 diabetes mellitus  ", ""])

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label."],
            cache_dir=tmp_path,
        )

    assert result == ["Obesity", "type 2 diabetes mellitus"]


async def test_list_approved_indications_skips_non_string_items(tmp_path):
    llm_response = json.dumps(["obesity", 42, None, "type 2 diabetes mellitus"])

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label."],
            cache_dir=tmp_path,
        )

    assert result == ["obesity", "type 2 diabetes mellitus"]


async def test_list_approved_indications_caches_result(tmp_path):
    llm_response = json.dumps(["obesity"])
    mock_llm = AsyncMock(return_value=llm_response)

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=mock_llm,
    ):
        first = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )
        second = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )

    assert first == ["obesity"]
    assert second == ["obesity"]
    mock_llm.assert_awaited_once()

