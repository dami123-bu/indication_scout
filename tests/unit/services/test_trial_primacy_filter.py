"""Unit tests for services.trial_primacy_filter.classify_trials_by_primacy."""

import logging
from unittest.mock import AsyncMock, patch

from indication_scout.models.model_clinical_trials import MeshTerm, Trial
from indication_scout.services.trial_primacy_filter import classify_trials_by_primacy

logger = logging.getLogger(__name__)


def _trial(nct_id: str, title: str = "T", indications=None, mesh=None) -> Trial:
    return Trial(
        nct_id=nct_id,
        title=title,
        indications=indications or [],
        mesh_conditions=[MeshTerm(id="", term=m) for m in (mesh or [])],
    )


async def test_classify_empty_input_returns_empty_set():
    """No trials → no LLM call → empty set."""
    result = await classify_trials_by_primacy(
        trials=[], queried_indication="cardiovascular disease", mesh_term="Cardiovascular Diseases"
    )
    assert result == set()


async def test_classify_returns_drop_set_from_llm_json():
    """Happy path: Haiku returns {'drop': [...]} and we surface that set."""
    trials = [
        _trial("NCT001", title="real CV trial"),
        _trial("NCT002", title="PCOS contraceptives — CV is just a covariate"),
    ]

    with patch(
        "indication_scout.services.trial_primacy_filter.query_small_llm",
        new=AsyncMock(return_value='{"drop": ["NCT002"]}'),
    ):
        result = await classify_trials_by_primacy(
            trials=trials,
            queried_indication="cardiovascular disease",
            mesh_term="Cardiovascular Diseases",
        )

    assert result == {"NCT002"}


async def test_classify_filters_out_hallucinated_nct_ids():
    """LLM returns an nct_id that isn't in the input — drop it from the result.

    Guards against the LLM inventing or misremembering NCT IDs.
    """
    trials = [_trial("NCT001"), _trial("NCT002")]

    with patch(
        "indication_scout.services.trial_primacy_filter.query_small_llm",
        new=AsyncMock(
            return_value='{"drop": ["NCT002", "NCT999_HALLUCINATED"]}'
        ),
    ):
        result = await classify_trials_by_primacy(
            trials=trials,
            queried_indication="cardiovascular disease",
            mesh_term="Cardiovascular Diseases",
        )

    assert result == {"NCT002"}


async def test_classify_returns_empty_on_llm_api_failure():
    """LLM raises → return empty set with a warning, preserving deterministic
    top-2 behavior. Better to keep all top-2 survivors than to drop everything.
    """
    trials = [_trial("NCT001"), _trial("NCT002")]

    with patch(
        "indication_scout.services.trial_primacy_filter.query_small_llm",
        new=AsyncMock(side_effect=RuntimeError("anthropic api error")),
    ):
        result = await classify_trials_by_primacy(
            trials=trials,
            queried_indication="cardiovascular disease",
            mesh_term="Cardiovascular Diseases",
        )

    assert result == set()


async def test_classify_returns_empty_on_unparseable_json():
    """LLM returns prose instead of JSON → empty set + warning. Keep all."""
    trials = [_trial("NCT001"), _trial("NCT002")]

    with patch(
        "indication_scout.services.trial_primacy_filter.query_small_llm",
        new=AsyncMock(return_value="I cannot determine this."),
    ):
        result = await classify_trials_by_primacy(
            trials=trials,
            queried_indication="cardiovascular disease",
            mesh_term="Cardiovascular Diseases",
        )

    assert result == set()


async def test_classify_returns_empty_when_json_missing_drop_key():
    """LLM returns JSON but no 'drop' list → empty set + warning."""
    trials = [_trial("NCT001"), _trial("NCT002")]

    with patch(
        "indication_scout.services.trial_primacy_filter.query_small_llm",
        new=AsyncMock(return_value='{"keep": ["NCT001"]}'),
    ):
        result = await classify_trials_by_primacy(
            trials=trials,
            queried_indication="cardiovascular disease",
            mesh_term="Cardiovascular Diseases",
        )

    assert result == set()


async def test_classify_handles_empty_drop_list():
    """LLM judges every trial PRIMARY → empty drop list → empty drop set."""
    trials = [_trial("NCT001"), _trial("NCT002")]

    with patch(
        "indication_scout.services.trial_primacy_filter.query_small_llm",
        new=AsyncMock(return_value='{"drop": []}'),
    ):
        result = await classify_trials_by_primacy(
            trials=trials,
            queried_indication="cardiovascular disease",
            mesh_term="Cardiovascular Diseases",
        )

    assert result == set()


async def test_classify_prompt_includes_indication_title_and_mesh():
    """Confirm the prompt the LLM receives carries the per-trial signal it
    needs: nct_id, title, sponsor-declared indications, and the top mesh
    terms — plus the queried indication and preferred MeSH term.
    """
    trials = [
        _trial(
            "NCT00842140",
            title="Safety of Hormonal Contraceptives in PCOS",
            indications=["Polycystic Ovary Syndrome"],
            mesh=["Polycystic Ovary Syndrome", "Cardiovascular Diseases"],
        ),
    ]
    captured_prompt = {}

    async def _capture(prompt: str, system: str = "") -> str:
        captured_prompt["prompt"] = prompt
        return '{"drop": ["NCT00842140"]}'

    with patch(
        "indication_scout.services.trial_primacy_filter.query_small_llm",
        new=AsyncMock(side_effect=_capture),
    ):
        await classify_trials_by_primacy(
            trials=trials,
            queried_indication="cardiovascular disease",
            mesh_term="Cardiovascular Diseases",
        )

    p = captured_prompt["prompt"]
    assert "NCT00842140" in p
    assert "Safety of Hormonal Contraceptives in PCOS" in p
    assert "Polycystic Ovary Syndrome" in p
    assert "Cardiovascular Diseases" in p
    assert "cardiovascular disease" in p
