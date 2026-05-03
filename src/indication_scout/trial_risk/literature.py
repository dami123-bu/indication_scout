"""Date-bounded literature signals for trial-risk features.

For each trial, we ask three questions of the PubMed corpus *as it stood
LOOKBACK_MONTHS before the trial's completion_date*:

  - failure_signal:  how strongly does the literature talk about failure /
                     termination / discontinuation of this drug for this disease?
  - safety_signal:   how strongly does it talk about safety / adverse events /
                     toxicity?
  - efficacy_signal: how strongly does it talk about efficacy / response /
                     positive outcomes?

Each signal is the mean cosine similarity of the top-k semantic-search hits
restricted to PMIDs whose `pub_date < cutoff_date`.

Reuses `services.retrieval.RetrievalService` for fetch+cache and
`services.embeddings.embed_async` for query embeddings. No new external calls
beyond what RetrievalService already does.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.embeddings import embed_async
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_MONTHS = 6
DEFAULT_TOP_K = 5
PUBMED_SLEEP_SECONDS = 5.0  # Conservative cushion against NCBI rate limits

FAILURE_QUERY = (
    "Clinical trial failure, termination, or discontinuation of {drug} for "
    "{disease}, including reasons such as lack of efficacy, safety concerns, "
    "recruitment failure, or toxicity."
)
SAFETY_QUERY = (
    "Adverse events, serious adverse events, toxicity, or safety concerns of "
    "{drug} in {disease}."
)
EFFICACY_QUERY = (
    "Clinical efficacy, treatment response, or positive outcomes of {drug} "
    "for {disease}."
)


@dataclass
class LiteratureSignals:
    """The three lit-derived features for a single (drug, disease) pair."""

    failure_signal: float = 0.0
    safety_signal: float = 0.0
    efficacy_signal: float = 0.0
    available: bool = False  # False if no PMIDs were found before the cutoff


def parse_trial_date(s: str | None) -> date | None:
    """Parse a ClinicalTrials.gov date (YYYY-MM-DD or YYYY-MM) to a date."""
    if not s:
        return None
    parts = s.split("-")
    try:
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
    except (ValueError, IndexError):
        return None
    return None


def cutoff_for_trial(trial: Trial, lookback_months: int) -> date | None:
    """Return `completion_date - lookback_months`, or None if completion_date missing."""
    end = parse_trial_date(trial.completion_date)
    if end is None:
        return None
    # Subtract months by going back month-by-month to avoid date arithmetic edge cases.
    year, month = end.year, end.month
    for _ in range(lookback_months):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    day = min(end.day, 28)  # safe day for any month
    return date(year, month, day)


async def _semantic_similarity_mean(
    query: str,
    pmids: list[str],
    cutoff_date: date,
    db: Session,
    top_k: int,
) -> float:
    """Embed `query`, find top-k hits among `pmids` with pub_date < cutoff, return mean cosine."""
    if not pmids:
        return 0.0

    query_vector = (await embed_async([query]))[0]
    rows = db.execute(
        text("""
            SELECT 1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM pubmed_abstracts
            WHERE pmid = ANY(:pmids)
              AND pub_date IS NOT NULL
              AND pub_date < :cutoff
            ORDER BY similarity DESC
            LIMIT :top_k
        """),
        {
            "query_vec": "[" + ",".join(str(x) for x in query_vector) + "]",
            "pmids": pmids,
            "cutoff": cutoff_date.isoformat(),
            "top_k": top_k,
        },
    ).fetchall()

    if not rows:
        return 0.0
    return sum(float(r[0]) for r in rows) / len(rows)


async def compute_signals(
    drug: str,
    disease: str,
    cutoff_date: date,
    db: Session,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    top_k: int = DEFAULT_TOP_K,
) -> LiteratureSignals:
    """Compute the three lit signals for one (drug, disease) pair at a cutoff.

    Fetches the broad PubMed PMID universe for `drug AND disease` (cached by
    pair, reused across trials and cutoffs), then runs semantic search against
    pgvector restricted to those PMIDs AND pub_date < cutoff_date. The date
    bound is enforced at the pgvector layer, not at PubMed — so the same
    PubMed query is reused across all trials sharing this (drug, disease)
    pair regardless of their individual cutoff dates.
    """
    if not drug or not disease:
        return LiteratureSignals()

    service = RetrievalService(cache_dir=cache_dir)

    try:
        chembl_id = await resolve_drug_name(drug, cache_dir)
        drug_aliases = await get_all_drug_names(chembl_id, cache_dir)
    except Exception as exc:
        logger.warning("Drug name expansion failed for %s: %s — using bare name", drug, exc)
        drug_aliases = [drug]

    queries = [f"{alias} AND {disease}" for alias in drug_aliases if alias]
    pmids: list[str] = []
    for q in queries:
        try:
            pair_pmids = await service.fetch_and_cache([q], db, date_before=None)
            pmids.extend(pair_pmids)
        except Exception as exc:
            logger.warning("PubMed fetch failed for %r: %s", q, exc)
        await asyncio.sleep(PUBMED_SLEEP_SECONDS)

    pmids = list(dict.fromkeys(pmids))  # order-preserving dedup
    if not pmids:
        return LiteratureSignals(available=False)

    failure = await _semantic_similarity_mean(
        FAILURE_QUERY.format(drug=drug, disease=disease),
        pmids, cutoff_date, db, top_k,
    )
    safety = await _semantic_similarity_mean(
        SAFETY_QUERY.format(drug=drug, disease=disease),
        pmids, cutoff_date, db, top_k,
    )
    efficacy = await _semantic_similarity_mean(
        EFFICACY_QUERY.format(drug=drug, disease=disease),
        pmids, cutoff_date, db, top_k,
    )
    return LiteratureSignals(
        failure_signal=failure,
        safety_signal=safety,
        efficacy_signal=efficacy,
        available=True,
    )


async def signals_for_trial(
    trial: Trial,
    drug: str,
    db: Session,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    top_k: int = DEFAULT_TOP_K,
) -> LiteratureSignals:
    """Compute lit signals for a trial, averaged across its mesh_conditions.

    Returns LiteratureSignals(available=False) when the trial has no
    completion_date or no mesh_conditions.
    """
    cutoff = cutoff_for_trial(trial, lookback_months)
    if cutoff is None:
        return LiteratureSignals()

    diseases = [m.term for m in trial.mesh_conditions if m.term]
    if not diseases:
        return LiteratureSignals()

    per_pair = []
    for disease in diseases:
        sig = await compute_signals(drug, disease, cutoff, db, cache_dir, top_k)
        if sig.available:
            per_pair.append(sig)

    if not per_pair:
        return LiteratureSignals(available=False)

    n = len(per_pair)
    return LiteratureSignals(
        failure_signal=sum(s.failure_signal for s in per_pair) / n,
        safety_signal=sum(s.safety_signal for s in per_pair) / n,
        efficacy_signal=sum(s.efficacy_signal for s in per_pair) / n,
        available=True,
    )
