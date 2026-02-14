"""Unit tests for base_client module."""

import pytest

from indication_scout.data_sources.base_client import BaseClient, DataSourceError


class ConcreteTestClient(BaseClient):
    """Concrete implementation of BaseClient for testing."""

    @property
    def _source_name(self) -> str:
        return "test_client"


@pytest.mark.asyncio
class TestBaseClient:
    """Unit tests for BaseClient session lifecycle (no network calls)."""

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
