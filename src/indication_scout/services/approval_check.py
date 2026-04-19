"""FDA approval check service.

Uses openFDA drug labels + LLM extraction to identify which candidate
diseases are already FDA-approved for a given drug's trade names.
"""

import json
import logging
from pathlib import Path
from typing import Any

from indication_scout.constants import CACHE_TTL, DEFAULT_CACHE_DIR
from indication_scout.data_sources.fda import FDAClient
from indication_scout.services.llm import query_small_llm, strip_markdown_fences
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

async def remove_approved_from_labels(
    drug_names: list[str],
    candidate_diseases: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> set[str]:
    """Trims the subset of candidate_diseases, remove ones that are already FDA-approved for the drug.

    Fetches openFDA label text for the drug names, then asks an LLM to remove
    any candidate already covered by the label's approved indications —
    including synonyms and clinically-contained narrower subsets. What remains
    are live repurposing candidates.

    Args:
        drug_names: Any drug names — trade names, generic/INN, USAN, etc.
                    (e.g. ["Ozempic", "Wegovy", "semaglutide"]).
        candidate_diseases: Disease names to filter.
        cache_dir: Cache directory for FDA label fetching.

    Returns:
        Set of candidate disease names (verbatim from input) that survived the
        filter — the live repurposing candidates. When labels are missing or
        the LLM response cannot be parsed, every candidate survives.
    """
    if not drug_names or not candidate_diseases:
        return set(candidate_diseases)

    async with FDAClient(cache_dir=cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_names)

    if not label_texts:
        return set(candidate_diseases)

    template = (_PROMPTS_DIR / "remove_fda_approvals.txt").read_text()
    prompt = template.format(
        label_texts="\n---\n".join(label_texts),
        candidate_diseases=", ".join(candidate_diseases),
    )

    response = await query_small_llm(prompt)
    stripped = strip_markdown_fences(response)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        logger.error(
            "remove_approved_from_labels: failed to parse LLM response: %s", response
        )
        return set(candidate_diseases)

    if not isinstance(parsed, list):
        logger.error(
            "remove_approved_from_labels: LLM returned non-list: %s", type(parsed)
        )
        return set(candidate_diseases)

    candidate_lower_map = {c.lower(): c for c in candidate_diseases}
    survivors: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        original = candidate_lower_map.get(item.lower())
        if original is not None:
            survivors.add(original)
        else:
            logger.warning(
                "remove_approved_from_labels: LLM returned unknown disease %r, skipping",
                item,
            )

    if len(survivors) > 1:
        survivors = await dedup_survivors(sorted(survivors))

    return survivors


async def dedup_survivors(diseases: list[str]) -> set[str]:
    """Collapse clinically-synonymous disease names into one entry per clinical entity.

    Asks an LLM to group entries that refer to the same clinical identity
    (e.g. "NAFLD" and "non-alcoholic fatty liver disease") and keep the full
    name over abbreviations.

    Args:
        diseases: Disease name strings to deduplicate.

    Returns:
        Set of kept disease names (verbatim from input). When the LLM
        response cannot be parsed, every input passes through unchanged.
    """
    if len(diseases) < 2:
        return set(diseases)

    template = (_PROMPTS_DIR / "dedup_diseases.txt").read_text()
    prompt = template.format(diseases=", ".join(diseases))

    response = await query_small_llm(prompt)
    stripped = strip_markdown_fences(response)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        logger.error("dedup_survivors: failed to parse LLM response: %s", response)
        return set(diseases)

    if not isinstance(parsed, list):
        logger.error("dedup_survivors: LLM returned non-list: %s", type(parsed))
        return set(diseases)

    input_lower_map = {d.lower(): d for d in diseases}
    kept: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        original = input_lower_map.get(item.lower())
        if original is not None:
            kept.add(original)
        else:
            logger.warning(
                "dedup_survivors: LLM returned unknown disease %r, skipping",
                item,
            )

    return kept


async def extract_approved_from_labels(
    label_texts: list[str],
    candidate_diseases: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> set[str]:
    """Use an LLM to identify which candidate diseases appear as approved indications in FDA label text.

    Args:
        label_texts: Raw indications_and_usage strings from openFDA.
        candidate_diseases: Disease names to check against the labels.
        cache_dir: Cache directory for storing LLM results.

    Returns:
        Set of candidate disease names (verbatim from input) found in the labels.
    """
    if not label_texts or not candidate_diseases:
        return set()

    cache_params = {
        "label_texts": sorted(label_texts),
        "candidate_diseases": sorted(candidate_diseases),
    }
    cached = cache_get("fda_approval_check", cache_params, cache_dir)
    if cached is not None:
        return set(cached)

    template = (_PROMPTS_DIR / "extract_fda_approvals.txt").read_text()
    prompt = template.format(
        label_texts="\n---\n".join(label_texts),
        candidate_diseases=", ".join(candidate_diseases),
    )

    response = await query_small_llm(prompt)

    stripped = strip_markdown_fences(response)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        logger.error(
            "extract_approved_from_labels: failed to parse LLM response: %s", response
        )
        return set()

    if not isinstance(parsed, list):
        logger.error(
            "extract_approved_from_labels: LLM returned non-list: %s", type(parsed)
        )
        return set()

    candidate_lower_map = {c.lower(): c for c in candidate_diseases}
    validated: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        original = candidate_lower_map.get(item.lower())
        if original is not None:
            validated.add(original)
        else:
            logger.warning(
                "extract_approved_from_labels: LLM returned unknown disease %r, skipping",
                item,
            )

    cache_set(
        "fda_approval_check",
        cache_params,
        list(validated),
        cache_dir,
        ttl=CACHE_TTL,
    )
    return validated

# TODO delete
async def get_all_fda_approved_diseases(
    drug_names: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Any:
    async with FDAClient(cache_dir=cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_names)

    if not label_texts:
        return set()
    return label_texts

async def get_fda_approved_diseases(
    drug_names: list[str],
    candidate_diseases: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> set[str]:
    """Fetch FDA labels for drug names and identify which candidates are already approved.

    Orchestrates FDAClient label fetching and LLM-based extraction.

    Args:
        drug_names: Any drug names — trade names, generic/INN, USAN, etc.
                    (e.g. ["Ozempic", "Wegovy", "semaglutide"]).
        candidate_diseases: Disease names to check.
        cache_dir: Cache directory.

    Returns:
        Set of candidate disease names that are FDA-approved according to labels.
    """
    if not drug_names or not candidate_diseases:
        return set()

    async with FDAClient(cache_dir=cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_names)

    if not label_texts:
        return set()

    return await extract_approved_from_labels(label_texts, candidate_diseases, cache_dir)
