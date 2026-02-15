"""Unit tests for OpenTargetsClient."""

from pathlib import Path


from indication_scout.data_sources.open_targets import OpenTargetsClient


# --- OpenTargetsClient configuration ---


def test_default_config():
    """Test that client uses default settings."""
    client = OpenTargetsClient()

    assert client.timeout == 30.0
    assert client.max_retries == 3
    assert client.cache_dir is not None
    assert client.cache_dir == Path("_cache")
