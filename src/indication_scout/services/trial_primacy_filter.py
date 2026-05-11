"""LLM filter for distinguishing primary-focus trials from adjacent-indication
leakage.

Background: CT.gov's `AREA[ConditionMeshTerm]` filter matches any trial whose
derived MeSH conditions include the queried descriptor, even when that
descriptor is a secondary/co-condition/population descriptor rather than the
trial's primary focus. The deterministic top-2 mesh-ordering filter in
clinical_trials_tools catches the obvious cases but cannot judge synonyms,
abbreviations, or semantic equivalence (e.g. NAFLD ↔ NASH, "Colon Cancer" vs
"Colorectal Neoplasms"). This module hands those judgment calls to Haiku.

The filter is intentionally invoked only on trials where the queried mesh is
at position 1+ — position-0 trials are deterministically primary by mesh
ordering and don't need LLM judgment.
"""

from __future__ import annotations

import logging

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.llm import parse_last_json_object, query_small_llm

logger = logging.getLogger(__name__)


def _trial_brief(trial: Trial) -> dict:
    """Compact per-trial payload sent to the LLM."""
    return {
        "nct_id": trial.nct_id,
        "title": trial.title,
        "indications": trial.indications,
        "mesh": [m.term for m in trial.mesh_conditions[:5]],
    }


async def classify_trials_by_primacy(
    trials: list[Trial],
    queried_indication: str,
    mesh_term: str,
) -> set[str]:
    """Classify each trial as PRIMARY or SECONDARY for `queried_indication`.

    Returns the set of nct_ids that the LLM judged SECONDARY (i.e. should be
    dropped). An empty input yields an empty set. A failed LLM call yields
    an empty set with a warning logged — callers treat that as "skip the
    filter" rather than "drop everything," preserving the deterministic
    top-2 behavior.

    The model only sees nct_id + title + sponsor-declared indications + the
    top-5 derived mesh terms. That's enough to judge primacy without leaking
    the LLM's own clinical priors into the decision.
    """
    if not trials:
        return set()

    briefs = [_trial_brief(t) for t in trials]
    prompt = _build_prompt(briefs, queried_indication, mesh_term)
    try:
        raw = await query_small_llm(prompt)
    except Exception as e:
        logger.warning(
            "trial primacy LLM call failed for indication=%r (%d trials): %r — "
            "skipping LLM filter, keeping all trials",
            queried_indication,
            len(trials),
            e,
        )
        return set()

    parsed = parse_last_json_object(raw)
    if parsed is None:
        logger.warning(
            "trial primacy LLM returned unparseable JSON for indication=%r: %r — "
            "skipping LLM filter, keeping all trials",
            queried_indication,
            raw[:200],
        )
        return set()

    drop_list = parsed.get("drop")
    if not isinstance(drop_list, list):
        logger.warning(
            "trial primacy LLM JSON missing 'drop' list for indication=%r: %r — "
            "skipping LLM filter, keeping all trials",
            queried_indication,
            parsed,
        )
        return set()

    valid_ids = {t.nct_id for t in trials}
    return {nct for nct in drop_list if isinstance(nct, str) and nct in valid_ids}


def _build_prompt(briefs: list[dict], queried_indication: str, mesh_term: str) -> str:
    """Build the classification prompt for Haiku."""
    rows = "\n".join(
        f"- {b['nct_id']}\n"
        f"    title: {b['title']!r}\n"
        f"    sponsor-declared indications: {b['indications']}\n"
        f"    derived mesh top-5: {b['mesh']}"
        for b in briefs
    )
    return f"""Task: For each trial below, decide whether the queried indication is the
trial's PRIMARY focus (the condition the trial is designed to study or treat)
or a SECONDARY tag (a co-condition, covariate, population descriptor, safety
context, or incidental MeSH enrichment).

Queried indication: {queried_indication!r}
Queried MeSH preferred term: {mesh_term!r}

Rules:
- PRIMARY = the trial's main therapeutic/disease focus is the queried indication,
  OR the queried indication is one of multiple co-primary focuses.
- SECONDARY = the queried indication appears only as a covariate, a population
  descriptor, a safety/risk context for a different primary focus, or as an
  inferred MeSH enrichment tag the sponsor didn't actually declare.
- Treat synonyms, abbreviations, and semantically equivalent phrasings as
  matching (e.g. NAFLD ↔ Non-alcoholic Steatohepatitis ↔ MASH; "T2DM" ↔
  "Type 2 Diabetes Mellitus"; "Colon Cancer" usually but not always ↔
  "Colorectal Cancer").
- Use the title and sponsor-declared indications as the strongest signal of
  primacy. Derived MeSH ordering is a hint, not authoritative.

Trials:
{rows}

Return a single JSON object with the nct_ids to DROP (i.e. classified as
SECONDARY):

{{"drop": ["NCT...", "NCT..."]}}

Return only the JSON object. Do not include any other text.
"""
