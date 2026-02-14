"""Integration tests for base_client module."""

import pytest

from indication_scout.data_sources.base_client import BaseClient, DataSourceError

pytestmark = pytest.mark.asyncio


class ConcreteTestClient(BaseClient):
    """Concrete implementation of BaseClient for testing."""

    @property
    def _source_name(self) -> str:
        return "test_client"


class TestBaseClient:
    """Integration tests for BaseClient (requires network)."""

    async def test_get_request_to_httpbin(self):
        """Test GET request to a real endpoint."""
        async with ConcreteTestClient(timeout=30.0) as client:
            result = await client._rest_get(
                "https://httpbin.org/get",
                params={"test_param": "test_value"},
            )

            assert result is not None
            assert result["args"]["test_param"] == "test_value"

    async def test_post_request_to_httpbin(self):
        """Test POST request to a real endpoint."""
        async with ConcreteTestClient(timeout=30.0) as client:
            result = await client._request(
                "POST",
                "https://httpbin.org/post",
                json_body={"key": "value"},
                headers={"Content-Type": "application/json"},
            )

            assert result is not None
            assert result["json"]["key"] == "value"

    async def test_timeout_raises_error(self):
        """Test that timeouts raise DataSourceError."""
        async with ConcreteTestClient(timeout=0.001, max_retries=0) as client:
            with pytest.raises(DataSourceError):
                await client._rest_get(
                    "https://httpbin.org/delay/10",  # 10 second delay
                    params={},
                )
