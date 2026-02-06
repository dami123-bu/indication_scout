"""Integration tests for base_client module."""

import asyncio
import json
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import aiohttp
import pytest

from indication_scout.data_sources.base_client import (
    BaseClient,
    CacheConfig,
    ClientConfig,
    DiskCache,
    PartialResult,
    RateLimitConfig,
    RequestContext,
    RetryConfig,
    TokenBucketRateLimiter,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter tests
# ---------------------------------------------------------------------------


class TestTokenBucketRateLimiter:
    """Integration tests for TokenBucketRateLimiter."""

    async def test_burst_allows_immediate_requests(self):
        """Test that burst requests are allowed immediately without delay."""
        config = RateLimitConfig(requests_per_second=10.0, burst=5)
        limiter = TokenBucketRateLimiter(config)

        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # All 5 requests should complete nearly instantly (burst)
        assert elapsed < 0.1

    async def test_rate_limiting_after_burst_exhausted(self):
        """Test that requests are delayed after burst is exhausted."""
        config = RateLimitConfig(requests_per_second=10.0, burst=2)
        limiter = TokenBucketRateLimiter(config)

        # Exhaust burst
        await limiter.acquire()
        await limiter.acquire()

        # Third request should be rate-limited
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should wait approximately 0.1s (1/10 requests per second)
        assert elapsed >= 0.05

    async def test_tokens_refill_over_time(self):
        """Test that tokens refill over time."""
        config = RateLimitConfig(requests_per_second=10.0, burst=2)
        limiter = TokenBucketRateLimiter(config)

        # Exhaust burst
        await limiter.acquire()
        await limiter.acquire()

        # Wait for tokens to refill
        await asyncio.sleep(0.3)

        # Should have refilled ~3 tokens, but capped at burst=2
        start = time.monotonic()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should be nearly instant since tokens refilled
        assert elapsed < 0.1

    async def test_concurrent_acquisitions(self):
        """Test that concurrent acquisitions are serialized properly."""
        config = RateLimitConfig(requests_per_second=100.0, burst=10)
        limiter = TokenBucketRateLimiter(config)

        # Launch many concurrent acquisitions
        tasks = [limiter.acquire() for _ in range(10)]
        await asyncio.gather(*tasks)

        # All should complete without errors (lock ensures serialization)
        # Just verify no exceptions were raised


# ---------------------------------------------------------------------------
# DiskCache tests
# ---------------------------------------------------------------------------


class TestDiskCache:
    """Integration tests for DiskCache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_cache_set_and_get(self, temp_cache_dir: Path):
        """Test basic cache set and get operations."""
        config = CacheConfig(enabled=True, directory=temp_cache_dir, ttl_seconds=3600)
        cache = DiskCache(config)

        namespace = "test_namespace"
        params = {"key1": "value1", "key2": 42}
        data = {"result": "some data", "count": 100}

        await cache.set(namespace, params, data)
        result = await cache.get(namespace, params)

        assert result == data

    async def test_cache_miss_returns_none(self, temp_cache_dir: Path):
        """Test that cache miss returns None."""
        config = CacheConfig(enabled=True, directory=temp_cache_dir, ttl_seconds=3600)
        cache = DiskCache(config)

        result = await cache.get("nonexistent", {"key": "value"})

        assert result is None

    async def test_cache_expiration(self, temp_cache_dir: Path):
        """Test that expired cache entries return None."""
        config = CacheConfig(enabled=True, directory=temp_cache_dir, ttl_seconds=1)
        cache = DiskCache(config)

        namespace = "test_expiry"
        params = {"key": "value"}
        data = {"result": "expires soon"}

        await cache.set(namespace, params, data)

        # Verify it's cached
        result = await cache.get(namespace, params)
        assert result == data

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired now
        result = await cache.get(namespace, params)
        assert result is None

    async def test_cache_disabled(self, temp_cache_dir: Path):
        """Test that disabled cache always returns None."""
        config = CacheConfig(enabled=False, directory=temp_cache_dir)
        cache = DiskCache(config)

        namespace = "test_disabled"
        params = {"key": "value"}
        data = {"result": "should not be cached"}

        await cache.set(namespace, params, data)
        result = await cache.get(namespace, params)

        assert result is None

    async def test_cache_invalidate(self, temp_cache_dir: Path):
        """Test cache invalidation removes entries."""
        config = CacheConfig(enabled=True, directory=temp_cache_dir, ttl_seconds=3600)
        cache = DiskCache(config)

        namespace = "test_invalidate"
        params = {"key": "value"}
        data = {"result": "to be invalidated"}

        await cache.set(namespace, params, data)
        result = await cache.get(namespace, params)
        assert result == data

        await cache.invalidate(namespace, params)
        result = await cache.get(namespace, params)
        assert result is None

    async def test_cache_different_params_different_keys(self, temp_cache_dir: Path):
        """Test that different params produce different cache keys."""
        config = CacheConfig(enabled=True, directory=temp_cache_dir, ttl_seconds=3600)
        cache = DiskCache(config)

        namespace = "test_keys"
        params1 = {"key": "value1"}
        params2 = {"key": "value2"}
        data1 = {"result": "data1"}
        data2 = {"result": "data2"}

        await cache.set(namespace, params1, data1)
        await cache.set(namespace, params2, data2)

        result1 = await cache.get(namespace, params1)
        result2 = await cache.get(namespace, params2)

        assert result1 == data1
        assert result2 == data2

    async def test_cache_custom_ttl(self, temp_cache_dir: Path):
        """Test that custom TTL overrides default."""
        config = CacheConfig(enabled=True, directory=temp_cache_dir, ttl_seconds=3600)
        cache = DiskCache(config)

        namespace = "test_custom_ttl"
        params = {"key": "value"}
        data = {"result": "short lived"}

        # Set with short TTL
        await cache.set(namespace, params, data, ttl=1)

        # Verify it's cached
        result = await cache.get(namespace, params)
        assert result == data

        # Wait for custom TTL to expire
        await asyncio.sleep(1.5)

        # Should be expired
        result = await cache.get(namespace, params)
        assert result is None

    async def test_cache_handles_corrupt_json(self, temp_cache_dir: Path):
        """Test that corrupt cache files are handled gracefully."""
        config = CacheConfig(enabled=True, directory=temp_cache_dir, ttl_seconds=3600)
        cache = DiskCache(config)

        namespace = "test_corrupt"
        params = {"key": "value"}

        # Manually write corrupt JSON
        key = cache._make_key(namespace, params)
        path = cache._path(key)
        path.write_text("not valid json {{{")

        # Should return None and not raise
        result = await cache.get(namespace, params)
        assert result is None


# ---------------------------------------------------------------------------
# BaseClient tests (using a concrete implementation)
# ---------------------------------------------------------------------------


class ConcreteTestClient(BaseClient):
    """Concrete implementation of BaseClient for testing."""

    @property
    def _source_name(self) -> str:
        return "test_client"


class TestBaseClient:
    """Integration tests for BaseClient."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_client_context_manager(self, temp_cache_dir: Path):
        """Test that client can be used as async context manager."""
        config = ClientConfig(
            cache=CacheConfig(enabled=True, directory=temp_cache_dir)
        )

        async with ConcreteTestClient(config) as client:
            assert client._session is None  # Session created lazily
            session = await client._get_session()
            assert session is not None
            assert not session.closed

        # Session should be closed after exiting context
        assert client._session.closed

    async def test_session_reuse(self, temp_cache_dir: Path):
        """Test that session is reused across requests."""
        config = ClientConfig(
            cache=CacheConfig(enabled=True, directory=temp_cache_dir)
        )
        client = ConcreteTestClient(config)

        session1 = await client._get_session()
        session2 = await client._get_session()

        assert session1 is session2
        await client.close()

    async def test_get_request_to_httpbin(self, temp_cache_dir: Path):
        """Test GET request to a real endpoint."""
        config = ClientConfig(
            cache=CacheConfig(enabled=False, directory=temp_cache_dir),
            timeout_seconds=30.0,
        )

        async with ConcreteTestClient(config) as client:
            result = await client._rest_get(
                "https://httpbin.org/get",
                params={"test_param": "test_value"},
                context=RequestContext(source="test", method="test_get"),
            )

            assert isinstance(result, PartialResult)
            assert result.is_complete is True
            assert result.data is not None
            assert result.data["args"]["test_param"] == "test_value"

    async def test_post_request_to_httpbin(self, temp_cache_dir: Path):
        """Test POST request to a real endpoint."""
        config = ClientConfig(
            cache=CacheConfig(enabled=False, directory=temp_cache_dir),
            timeout_seconds=30.0,
        )

        async with ConcreteTestClient(config) as client:
            result = await client._request(
                "POST",
                "https://httpbin.org/post",
                json_body={"key": "value"},
                headers={"Content-Type": "application/json"},
                context=RequestContext(source="test", method="test_post"),
            )

            assert isinstance(result, PartialResult)
            assert result.is_complete is True
            assert result.data is not None
            assert result.data["json"]["key"] == "value"

    async def test_caching_works(self, temp_cache_dir: Path):
        """Test that responses are cached and returned from cache."""
        config = ClientConfig(
            cache=CacheConfig(enabled=True, directory=temp_cache_dir, ttl_seconds=3600),
            timeout_seconds=30.0,
        )

        async with ConcreteTestClient(config) as client:
            # First request - should hit network
            result1 = await client._rest_get(
                "https://httpbin.org/get",
                params={"cache_test": "1"},
                cache_namespace="test_cache",
                context=RequestContext(source="test", method="test_cache"),
            )

            assert result1.cached is False
            assert result1.is_complete is True

            # Second request - should hit cache
            result2 = await client._rest_get(
                "https://httpbin.org/get",
                params={"cache_test": "1"},
                cache_namespace="test_cache",
                context=RequestContext(source="test", method="test_cache"),
            )

            assert result2.cached is True
            assert result2.data == result1.data

    async def test_timeout_handling(self, temp_cache_dir: Path):
        """Test that timeouts are handled gracefully."""
        config = ClientConfig(
            cache=CacheConfig(enabled=False, directory=temp_cache_dir),
            timeout_seconds=0.001,  # Very short timeout
            retry=RetryConfig(max_retries=0),  # No retries
        )

        async with ConcreteTestClient(config) as client:
            result = await client._rest_get(
                "https://httpbin.org/delay/10",  # 10 second delay
                params={},
                context=RequestContext(source="test", method="test_timeout"),
            )

            assert result.is_complete is False
            assert result.data is None
            assert len(result.errors) > 0

    async def test_apply_date_filter(self):
        """Test date filter application."""
        params = {"existing": "value"}
        date_before = date(2024, 1, 15)

        result = BaseClient.apply_date_filter(
            params,
            date_before,
            date_param_name="date_cutoff",
            date_format="%Y-%m-%d",
        )

        assert result["existing"] == "value"
        assert result["date_cutoff"] == "2024-01-15"

    async def test_apply_date_filter_none(self):
        """Test date filter with None date."""
        params = {"existing": "value"}

        result = BaseClient.apply_date_filter(
            params,
            None,
            date_param_name="date_cutoff",
        )

        assert result == params
        assert "date_cutoff" not in result


# ---------------------------------------------------------------------------
# RequestContext tests
# ---------------------------------------------------------------------------


class TestRequestContext:
    """Tests for RequestContext model."""

    def test_request_context_creation(self):
        """Test RequestContext can be created with required fields."""
        ctx = RequestContext(
            source="test_source",
            method="test_method",
        )

        assert ctx.source == "test_source"
        assert ctx.method == "test_method"
        assert ctx.agent is None
        assert ctx.params == {}

    def test_request_context_with_optional_fields(self):
        """Test RequestContext with all fields."""
        ctx = RequestContext(
            source="test_source",
            method="test_method",
            agent="biology",
            params={"key": "value"},
        )

        assert ctx.source == "test_source"
        assert ctx.method == "test_method"
        assert ctx.agent == "biology"
        assert ctx.params == {"key": "value"}


# ---------------------------------------------------------------------------
# PartialResult tests
# ---------------------------------------------------------------------------


class TestPartialResult:
    """Tests for PartialResult model."""

    def test_partial_result_defaults(self):
        """Test PartialResult default values."""
        result = PartialResult(data={"key": "value"})

        assert result.data == {"key": "value"}
        assert result.is_complete is True
        assert result.errors == []
        assert result.cached is False
        assert result.elapsed_seconds == 0.0

    def test_partial_result_incomplete(self):
        """Test PartialResult for incomplete response."""
        result = PartialResult(
            data=None,
            is_complete=False,
            errors=["Timeout occurred"],
            elapsed_seconds=5.0,
        )

        assert result.data is None
        assert result.is_complete is False
        assert result.errors == ["Timeout occurred"]
        assert result.elapsed_seconds == 5.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfigs:
    """Tests for configuration models."""

    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.backoff_factor == 2.0
        assert 429 in config.retryable_status_codes

    def test_rate_limit_config_defaults(self):
        """Test RateLimitConfig default values."""
        config = RateLimitConfig()

        assert config.requests_per_second == 5.0
        assert config.burst == 10

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.ttl_seconds == 86400

    def test_client_config_composition(self):
        """Test ClientConfig composes other configs."""
        config = ClientConfig(
            retry=RetryConfig(max_retries=5),
            rate_limit=RateLimitConfig(requests_per_second=10.0),
            cache=CacheConfig(enabled=False),
            timeout_seconds=60.0,
        )

        assert config.retry.max_retries == 5
        assert config.rate_limit.requests_per_second == 10.0
        assert config.cache.enabled is False
        assert config.timeout_seconds == 60.0