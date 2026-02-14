"""Unit tests for OpenTargetsClient."""

import hashlib
import json
from pathlib import Path

from indication_scout.data_sources.open_targets import OpenTargetsClient


class TestOpenTargetsClientConfig:
    """Tests for OpenTargetsClient configuration."""

    def test_default_config(self):
        """Test that client uses default settings."""
        client = OpenTargetsClient()

        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.cache_dir == Path("_cache")


class TestCacheKeyGeneration:
    """Tests for _cache_key determinism and uniqueness."""

    def setup_method(self) -> None:
        self.client = OpenTargetsClient()

    def test_deterministic_same_inputs(self) -> None:
        """Same namespace + params always produce the same key."""
        key1 = self.client._cache_key("drug", {"chembl_id": "CHEMBL25"})
        key2 = self.client._cache_key("drug", {"chembl_id": "CHEMBL25"})

        assert key1 == key2

    def test_different_namespace_different_key(self) -> None:
        """Different namespaces with identical params produce different keys."""
        key_drug = self.client._cache_key("drug", {"id": "CHEMBL25"})
        key_target = self.client._cache_key("target", {"id": "CHEMBL25"})

        assert key_drug != key_target

    def test_different_params_different_key(self) -> None:
        """Same namespace with different params produce different keys."""
        key1 = self.client._cache_key("drug", {"chembl_id": "CHEMBL25"})
        key2 = self.client._cache_key("drug", {"chembl_id": "CHEMBL1431"})

        assert key1 != key2

    def test_param_order_does_not_affect_key(self) -> None:
        """Dict key ordering should not change the hash (sort_keys=True)."""
        key1 = self.client._cache_key("drug", {"a": "1", "b": "2"})
        key2 = self.client._cache_key("drug", {"b": "2", "a": "1"})

        assert key1 == key2

    def test_key_is_valid_sha256_hex(self) -> None:
        """Key should be a 64-character lowercase hex string (SHA-256)."""
        key = self.client._cache_key("drug", {"chembl_id": "CHEMBL25"})

        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_matches_expected_sha256(self) -> None:
        """Key matches a manually computed SHA-256 of the canonical JSON."""
        namespace = "drug"
        params = {"chembl_id": "CHEMBL25"}
        raw = json.dumps({"ns": namespace, **params}, sort_keys=True, default=str)
        expected = hashlib.sha256(raw.encode()).hexdigest()

        key = self.client._cache_key(namespace, params)

        assert key == expected
