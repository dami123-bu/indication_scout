"""Phase 5 regression: pre- vs post-MeSH-filter trial counts.

Hits NCBI (MeSH resolver) and ClinicalTrials.gov for a small, fixed set of
drug-indication pairs. For each pair, runs `search_trials` once without the
MeSH filter (Essie recall — includes glaucoma, portal, pulmonary hypertension
when querying "hypertension") and once with the resolved MeSH D-number, then
logs the pre/post counts so the human can eyeball whether the filter is
narrowing too aggressively.

`is_whitespace` is `exact_match_count == 0`, so there is no numeric threshold
to recalibrate against. The assertions instead check filter sanity:

  - post <= pre   (the filter only narrows; never invents trials)
  - at least 2/3  pairs retain >0 trials (filter isn't catastrophically
                  over-dropping known-active pairs)

A `date_before=2025-01-01` cutoff is used to keep counts roughly stable across
runs; counts may still drift as CT.gov re-tags trials.
"""

import logging
from datetime import date

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.services.disease_helper import resolve_mesh_id

logger = logging.getLogger(__name__)

_CUTOFF = date(2025, 1, 1)

# (drug, indication) — chosen because the indication has known Essie noise
# (sibling MeSH branches the loose CT.gov query mixes in).
_PAIRS: list[tuple[str, str]] = [
    ("metformin", "hypertension"),
    ("aspirin", "diabetes mellitus"),
    ("semaglutide", "hypertension"),
]


async def test_mesh_filter_pre_vs_post_counts(clinical_trials_client: ClinicalTrialsClient):
    results: list[tuple[str, str, str, int, int]] = []

    for drug, indication in _PAIRS:
        mesh_id = await resolve_mesh_id(indication)
        assert mesh_id is not None, f"resolver returned None for {indication!r}"

        pre_trials = await clinical_trials_client.search_trials(
            drug=drug, indication=indication, date_before=_CUTOFF
        )
        post_trials = await clinical_trials_client.search_trials(
            drug=drug,
            indication=indication,
            date_before=_CUTOFF,
            target_mesh_id=mesh_id,
        )

        pre, post = len(pre_trials), len(post_trials)
        results.append((drug, indication, mesh_id, pre, post))

        logger.info(
            "MeSH filter: %s x %s (mesh=%s) — pre=%d post=%d dropped=%d",
            drug,
            indication,
            mesh_id,
            pre,
            post,
            pre - post,
        )

        # Filter only narrows.
        assert post <= pre, (
            f"{drug} x {indication}: post-filter count {post} exceeds pre-filter {pre}"
        )

    # Most pairs should retain trials. If they all collapse to zero, the
    # filter is over-aggressive (or the resolver picked the wrong MeSH
    # branch). Threshold scales with the active pair list so commenting
    # pairs in/out doesn't break the assertion.
    nonempty = sum(1 for *_, post in results if post > 0)
    min_nonempty = max(1, len(results) - 1)
    assert nonempty >= min_nonempty, (
        f"only {nonempty}/{len(results)} pairs retained trials post-filter: {results}"
    )
