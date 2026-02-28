"""Unit tests for PubMedClient caching."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.constants import CACHE_TTL
from indication_scout.data_sources.pubmed import PubMedClient


def test_default_config():
    client = PubMedClient()
    assert client.cache_dir == Path("_cache")


def test_cache_disabled(tmp_path):
    client = PubMedClient(cache_dir=None)
    result = client._cache_get("pubmed_search", {"query": "metformin"})
    assert result is None


def test_cache_miss_then_hit(tmp_path):
    client = PubMedClient(cache_dir=tmp_path)
    params = {"query": "metformin", "max_results": 10, "date_before": None}

    assert client._cache_get("pubmed_search", params) is None

    client._cache_set("pubmed_search", params, ["11111111", "22222222"])
    result = client._cache_get("pubmed_search", params)
    assert result == ["11111111", "22222222"]


def test_cache_expired(tmp_path):
    client = PubMedClient(cache_dir=tmp_path)
    params = {"query": "aspirin", "max_results": 5, "date_before": None}

    # Write a cache entry with cached_at far in the past
    expired_at = (datetime.now() - timedelta(seconds=CACHE_TTL + 1)).isoformat()
    key = client._cache_key("pubmed_search", params)
    path = client._cache_path(key)
    path.write_text(
        json.dumps({"data": ["99999999"], "cached_at": expired_at, "ttl": CACHE_TTL})
    )

    result = client._cache_get("pubmed_search", params)
    assert result is None
    assert not path.exists()


async def test_search_uses_cache(tmp_path):
    client = PubMedClient(cache_dir=tmp_path)
    mock_response = {"esearchresult": {"idlist": ["12345678", "87654321"]}}

    with patch.object(
        client, "_rest_get", new=AsyncMock(return_value=mock_response)
    ) as mock_get:
        result1 = await client.search("metformin diabetes", max_results=10)
        result2 = await client.search("metformin diabetes", max_results=10)

    assert result1 == ["12345678", "87654321"]
    assert result2 == ["12345678", "87654321"]
    mock_get.assert_called_once()
