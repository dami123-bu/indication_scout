"""
Disease term normalizer for PubMed search.

Converts raw disease terms from Open Targets (e.g., "narcolepsy-cataplexy syndrome")
into normalized PubMed-friendly search terms (e.g., "narcolepsy").

Strategy: LLM normalize → verify with PubMed count → cache everything.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from indication_scout.constants import CACHE_TTL, DEFAULT_CACHE_DIR
from indication_scout.services.llm import (
    query_small_llm,
)  # Adjust import path as needed

logger = logging.getLogger(__name__)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MIN_RESULTS = 3  # Minimum PubMed hits to consider a term useful

# Terms that are too generic to be useful as a broadened disease search term.
# If the LLM fallback generalizes to one of these, we discard it and keep the
# more specific first result.
BROADENING_BLOCKLIST: frozenset[str] = frozenset(
    {
        "cancer",
        "tumor",
        "tumour",
        "neoplasm",
        "malignancy",
        "disease",
        "disorder",
        "syndrome",
        "condition",
    }
)

NORMALIZE_PROMPT = (
    "Convert this disease term into a PubMed search term. "
    "Remove subtypes, staging, etiology, and genetic qualifiers, but KEEP the organ or tissue specificity. "
    "Do NOT generalize to a broader disease class (e.g. do not map 'lung neoplasm' to 'cancer'). "
    "If the disease has a well-known common-name synonym, include both joined with OR.\n\n"
    "Examples:\n"
    "atopic eczema → eczema OR dermatitis\n"
    "narcolepsy-cataplexy syndrome → narcolepsy\n"
    "non-small cell lung carcinoma → lung cancer OR lung neoplasm\n"
    "hereditary hemorrhagic telangiectasia → telangiectasia\n"
    "myocardial infarction → heart attack OR myocardial infarction\n"
    "smoking cessation → smoking OR nicotine dependence\n"
    "portal hypertension → hypertension\n"
    "renal tubular dysgenesis → kidney disease\n"
    "hepatocellular carcinoma → liver cancer OR hepatocellular carcinoma\n\n"
    "Return ONLY the term(s), nothing else.\n\n"
    "Term: {raw_term}"
)


# ── Cache ────────────────────────────────────────────────────────────────────


def _cache_key(namespace: str, params: dict[str, Any]) -> str:
    raw = json.dumps({"ns": namespace, **params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(
    namespace: str, params: dict[str, Any], cache_dir: Path = DEFAULT_CACHE_DIR
) -> Any | None:
    path = cache_dir / f"{_cache_key(namespace, params)}.json"
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text())
        age = (
            datetime.now() - datetime.fromisoformat(entry["cached_at"])
        ).total_seconds()
        if age > entry.get("ttl", CACHE_TTL):
            path.unlink(missing_ok=True)
            return None
        return entry["data"]
    except (json.JSONDecodeError, KeyError, ValueError):
        path.unlink(missing_ok=True)
        return None


def _cache_set(
    namespace: str,
    params: dict[str, Any],
    data: Any,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = {"data": data, "cached_at": datetime.now().isoformat(), "ttl": CACHE_TTL}
    (cache_dir / f"{_cache_key(namespace, params)}.json").write_text(
        json.dumps(entry, default=str)
    )


# ── LLM Normalize ───────────────────────────────────────────────────────────


async def llm_normalize_disease(raw_term: str) -> str:
    """
    Reduce disease to overarching

    Use an LLM to convert a disease term into a PubMed search term.

    Example:
        "narcolepsy-cataplexy syndrome"         → "narcolepsy"
        "renal tubular dysgenesis"              → "kidney disease"
        "CML"                                   → "chronic myeloid leukemia"
    """
    cached = _cache_get("disease_norm", {"raw_term": raw_term})
    if cached is not None:
        return cached

    prompt = NORMALIZE_PROMPT.format(raw_term=raw_term)
    response = await query_small_llm(prompt)
    normalized = response.strip().strip('"').strip("'")

    _cache_set("disease_norm", {"raw_term": raw_term}, normalized)
    return normalized


# ── PubMed Count ─────────────────────────────────────────────────────────────


async def pubmed_count(query: str) -> int:
    """Return the number of PubMed results for a query string."""
    cached = _cache_get("pubmed_count", {"query": query})
    if cached is not None:
        return cached

    url = f"{NCBI_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 0,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.warning(f"PubMed count failed for '{query}': {e}")
            return 0

    count = int(data.get("esearchresult", {}).get("count", 0))
    _cache_set("pubmed_count", {"query": query}, count)
    return count


# ── Main Orchestrator ────────────────────────────────────────────────────────


async def normalize_for_pubmed(raw_term: str, drug_name: Optional[str] = None) -> str:
    """
    Short set of queries
    e.g. hepatocellular carcinoma -> 'liver cancer OR hepatocellular carcinoma'

    Normalize a raw disease term from Open Targets into a list of PubMed queries.

    Strategy:
        1. Check cache
        2. LLM normalize
        3. If drug_name provided, verify with PubMed count (drug + disease)
        4. If too few results, ask LLM to generalize further
        5. Cache and return

    Args:
        raw_term:  Disease term from Open Targets (e.g., "narcolepsy-cataplexy syndrome")
        drug_name: Optional drug being investigated (e.g., "bupropion") — used for verification.
                   If None, skips PubMed count verification and returns LLM result directly.

    Returns:
        Normalized PubMed search term (e.g., "narcolepsy")
    """
    # Step 1: LLM normalize
    normalized = await llm_normalize_disease(raw_term)

    # Reject if LLM collapsed to a blocklisted over-generic term
    normalized_terms = {t.strip().lower() for t in normalized.split("OR")}
    if normalized_terms <= BROADENING_BLOCKLIST:
        logger.info(
            f"Rejected over-broad normalization '{normalized}' for '{raw_term}', keeping raw term"
        )
        normalized = raw_term

    # Step 2: If drug provided, verify with PubMed count
    if drug_name:
        count = await pubmed_count(f"{drug_name} AND ({normalized})")

        if count < MIN_RESULTS:
            # Step 3: Ask LLM to generalize further
            broader = await llm_normalize_disease(
                f"{normalized} (generalize to a broader disease category)"
            )
            broader_terms = {t.strip().lower() for t in broader.split("OR")}
            if broader_terms & BROADENING_BLOCKLIST:
                logger.info(
                    f"Rejected over-broad fallback '{broader}' for '{normalized}'"
                )
            else:
                broader_count = await pubmed_count(f"{drug_name} AND ({broader})")
                if broader_count >= MIN_RESULTS:
                    normalized = broader

    logger.info(f"Normalized '{raw_term}' → '{normalized}'")

    return normalized


# ── Batch Convenience ────────────────────────────────────────────────────────


async def normalize_batch(
    terms: list[str], drug_name: Optional[str] = None
) -> dict[str, str]:
    """
    Normalize a list of disease terms. Returns dict of raw → normalized.
    Processes sequentially to respect NCBI rate limits (3 req/sec without API key,
    10 req/sec with one).
    """
    results = {}
    for term in terms:
        results[term] = await normalize_for_pubmed(term, drug_name)
        await asyncio.sleep(0.35)  # Stay under NCBI rate limit

    return results
