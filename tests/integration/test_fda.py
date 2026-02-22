"""Integration tests for FDAClient."""

import pytest

from indication_scout.constants import REACTION_OUTCOME_MAP
from indication_scout.data_sources.base_client import DataSourceError


# --- Main functionality ---


@pytest.mark.asyncio
async def test_get_top_reactions(fda_client):
    """Test get_top_reactions returns reaction counts for a known drug."""
    reactions = await fda_client.get_top_reactions("metformin", limit=5)

    assert len(reactions) == 5

    # Extract first reaction and verify all fields
    first = reactions[0]
    assert isinstance(first.term, str)
    assert first.term != ""
    assert first.count >= 1

    # Results are ordered by count descending
    for i in range(len(reactions) - 1):
        assert reactions[i].count >= reactions[i + 1].count


@pytest.mark.asyncio
async def test_get_events(fda_client):
    """Test get_events returns event records for a known drug."""
    events = await fda_client.get_events("metformin", limit=5)

    assert len(events) == 5

    # Extract first event and verify all 6 fields
    first = events[0]
    assert isinstance(first.medicinal_product, str)
    assert first.medicinal_product != ""
    assert isinstance(first.reaction, str)
    assert first.reaction != ""
    # drug_indication may or may not be present
    assert first.drug_indication is None or isinstance(first.drug_indication, str)
    # reaction_outcome is mapped from numeric code or None
    valid_outcomes = set(REACTION_OUTCOME_MAP.values())
    assert first.reaction_outcome is None or first.reaction_outcome in valid_outcomes
    # serious is "1" or "2" or None
    assert first.serious in ("1", "2", None)
    # company_numb is a string or None
    assert first.company_numb is None or isinstance(first.company_numb, str)


@pytest.mark.asyncio
async def test_get_events_has_outcome(fda_client):
    """Test that at least some events have a mapped reaction outcome."""
    events = await fda_client.get_events("metformin", limit=10)

    valid_outcomes = set(REACTION_OUTCOME_MAP.values())
    outcomes = [e.reaction_outcome for e in events if e.reaction_outcome is not None]
    # Metformin has millions of FAERS reports; 10 events should include at least one outcome
    assert len(outcomes) >= 1
    for outcome in outcomes:
        assert outcome in valid_outcomes


# --- Edge cases ---


@pytest.mark.asyncio
async def test_get_top_reactions_nonexistent_drug(fda_client):
    """Test that a nonexistent drug raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await fda_client.get_top_reactions("xyzzy_fake_drug_99999_not_real")

    assert exc_info.value.source == "openfda"


@pytest.mark.asyncio
async def test_get_events_nonexistent_drug(fda_client):
    """Test that a nonexistent drug raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await fda_client.get_events("xyzzy_fake_drug_99999_not_real")

    assert exc_info.value.source == "openfda"


@pytest.mark.asyncio
async def test_get_top_reactions_limit_respected(fda_client):
    """Test that the limit parameter controls how many results are returned."""
    reactions = await fda_client.get_top_reactions("metformin", limit=3)

    assert len(reactions) == 3
