"""Unit tests for the literature client factory."""

import pytest

from indication_scout.data_sources.europe_pmc import EuropePMCClient
from indication_scout.data_sources.literature import get_literature_client
from indication_scout.data_sources.pubmed import PubMedClient


@pytest.mark.parametrize(
    "source, expected_cls",
    [
        ("pubmed", PubMedClient),
        ("europe_pmc", EuropePMCClient),
    ],
)
def test_factory_returns_expected_client(monkeypatch, source, expected_cls):
    from indication_scout import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("LITERATURE_SOURCE", source)
    try:
        client = get_literature_client()
        assert isinstance(client, expected_cls)
    finally:
        config.get_settings.cache_clear()


def test_factory_rejects_unknown_source(monkeypatch):
    from indication_scout import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("LITERATURE_SOURCE", "wikipedia")
    try:
        with pytest.raises(ValueError, match="Unknown literature_source"):
            get_literature_client()
    finally:
        config.get_settings.cache_clear()
