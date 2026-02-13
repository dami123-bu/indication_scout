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
    """Integration tests for BaseClient."""

    async def test_client_context_manager(self):
        """Test that client can be used as async context manager."""
        async with ConcreteTestClient() as client:
            assert client._session is None  # Session created lazily
            session = await client._get_session()
            assert session is not None
            assert not session.closed

        # Session should be closed after exiting context
        assert client._session.closed

    async def test_session_reuse(self):
        """Test that session is reused across requests."""
        client = ConcreteTestClient()

        session1 = await client._get_session()
        session2 = await client._get_session()

        assert session1 is session2
        await client.close()

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


class TestDataSourceError:
    """Tests for DataSourceError."""

    def test_error_message_format(self):
        """Test error message includes source."""
        error = DataSourceError("pubmed", "Connection failed")
        assert "[pubmed]" in str(error)
        assert "Connection failed" in str(error)

    def test_error_with_status_code(self):
        """Test error can include status code."""
        error = DataSourceError("api", "Not found", status_code=404)
        assert error.source == "api"
        assert error.status_code == 404
