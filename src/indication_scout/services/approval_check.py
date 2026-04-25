"""FDA approval check service.

Two paths are exposed:

  - get_fda_approved_disease_mapping (LEGACY, openFDA-based)
    Resolves the drug to all known aliases via ChEMBL, fetches all matching
    openFDA drug labels, and asks the LLM to extract approvals from the
    label text. Suffers from openFDA top-N truncation, bug-named indications,
    and rate limits. Kept for backward compatibility; will be removed.

  - get_ot_approved_disease_mapping (NEW, Open Targets-based)
    Searches Open Targets for the drug, picks the best-matching ChEMBL
    entry (preferring exact-name match), pulls the APPROVED-stage
    indication list, and asks the LLM to bridge candidates to that clean
    list (synonym/parent/lay-term judgment only — no label parsing).
    Cleaner data, simpler prompt, faster, no rate limits.

Both functions consult CURATED_FDA_APPROVED_CANDIDATES (force True) and
CURATED_FDA_REJECTED_CANDIDATES (force False) before invoking the LLM.
"""

import json
import logging
from pathlib import Path
from typing import Any

from indication_scout.constants import (
    CACHE_TTL,
    CURATED_FDA_APPROVED_CANDIDATES,
    CURATED_FDA_REJECTED_CANDIDATES,
    DEFAULT_CACHE_DIR,
    OPEN_TARGETS_BASE_URL,
)
from indication_scout.data_sources.chembl import (
    get_all_drug_names,
    resolve_drug_name,
)
from indication_scout.data_sources.fda import FDAClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
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

async def get_fda_approved_disease_mapping(
    drug_name: str,
    candidate_diseases: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, bool]:
    """Decide, per candidate, whether the drug is FDA-approved for that disease.

    Two-tier lookup:
      1. Curated short-circuit — if a candidate string EXACTLY matches
         (case-sensitive) any string in CURATED_FDA_APPROVED_CANDIDATES[drug_name],
         it returns True without calling the LLM.
      2. LLM fallback — for remaining candidates, the input drug_name is
         expanded to all known aliases (generic, trade, INN, USAN, salt forms)
         via ChEMBL, all matching openFDA labels are fetched, and the
         candidates are batched into one LLM call that returns true/false
         per candidate.

    Args:
        drug_name: A single drug name (trade, generic/INN, or USAN).
        candidate_diseases: Disease names to check against the label.
        cache_dir: Cache directory.

    Returns:
        Dict mapping each input candidate (verbatim) to True if the drug is
        FDA-approved for it, False otherwise. Every input candidate is always
        present as a key. Defaults to False on any failure (chembl, FDA fetch,
        LLM parse).
    """
    result: dict[str, bool] = {c: False for c in candidate_diseases}

    if not drug_name or not candidate_diseases:
        return result

    # Curated short-circuit: exact, case-sensitive match against the drug's
    # curated approved-candidate list. Hits skip both the FDA fetch and the
    # LLM call.
    curated_set = set(CURATED_FDA_APPROVED_CANDIDATES.get(drug_name, []))
    uncurated: list[str] = []
    for c in candidate_diseases:
        if c in curated_set:
            result[c] = True
        else:
            uncurated.append(c)

    if not uncurated:
        return result

    # Expand the input drug name to all known aliases (generic, trade, INN,
    # USAN, salt forms, etc.) via ChEMBL. Different formulations of the same
    # drug carry distinct openFDA labels (e.g. fluoxetine generic vs Sarafem
    # for PMDD), so feeding all aliases to get_all_label_indications surfaces
    # approvals that a single-name lookup would miss. On any chembl failure,
    # fall back to the bare drug_name.
    try:
        chembl_id = await resolve_drug_name(drug_name, cache_dir)
        drug_aliases = await get_all_drug_names(chembl_id, cache_dir)
        if drug_name not in drug_aliases:
            drug_aliases = [drug_name, *drug_aliases]
        logger.info(
            "get_fda_approved_disease_mapping: %r → chembl_id=%s, %d aliases",
            drug_name, chembl_id, len(drug_aliases),
        )
    except Exception as e:
        logger.warning(
            "get_fda_approved_disease_mapping: chembl alias lookup failed for %r: %s; "
            "falling back to bare drug_name",
            drug_name, e,
        )
        drug_aliases = [drug_name]

    async with FDAClient(cache_dir=cache_dir) as client:
        label_texts = await client.get_all_label_indications(drug_aliases)

    logger.info(
        "get_fda_approved_disease_mapping: %r → fetched %d label texts from %d aliases",
        drug_name, len(label_texts), len(drug_aliases),
    )

    if not label_texts:
        logger.warning(
            "get_fda_approved_disease_mapping: %r → no label texts found across %d aliases; "
            "returning False for %d candidate(s)",
            drug_name, len(drug_aliases), len(uncurated),
        )
        return result

    template = (_PROMPTS_DIR / "extract_fda_approval_single.txt").read_text()
    prompt = template.format(
        label_texts="\n---\n".join(label_texts),
        candidate_diseases=json.dumps(uncurated),
    )

    response = await query_small_llm(prompt)
    stripped = strip_markdown_fences(response)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        logger.error(
            "get_fda_approved_disease_mapping: failed to parse LLM response: %s",
            response,
        )
        return result

    if not isinstance(parsed, dict):
        logger.error(
            "get_fda_approved_disease_mapping: LLM returned non-dict: %s",
            type(parsed),
        )
        return result

    # Map LLM keys back to verbatim input candidates (case-insensitive), scoped
    # to uncurated candidates so a stray key cannot overwrite a curated value.
    lower_to_verbatim = {c.lower(): c for c in uncurated}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        original = lower_to_verbatim.get(key.lower())
        if original is None:
            logger.warning(
                "get_fda_approved_disease_mapping: LLM returned unknown candidate %r, skipping",
                key,
            )
            continue
        if not isinstance(value, bool):
            logger.error(
                "get_fda_approved_disease_mapping: value for %r is not a bool: %s",
                key, type(value),
            )
            continue
        result[original] = value

    return result


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


# -- Open Targets-based path --------------------------------------------------
# New approval-check path. Searches OT for the drug, picks the best-matching
# ChEMBL entry by name + APPROVAL count, pulls the APPROVED-stage indication
# list, and asks the LLM to bridge candidates to that clean list. Same input/
# output contract as get_fda_approved_disease_mapping.

_OT_DRUG_SEARCH_QUERY = """
query($q: String!) {
    search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 10}) {
        hits { id name entity }
    }
}
"""


async def _ot_pick_drug_and_fetch_approved(
    drug_name: str, cache_dir: Path
) -> tuple[str | None, list[str]]:
    """Search OT for the drug, return (chembl_id, approved_disease_names).

    Prefers hits whose pref_name matches the query (case-insensitive); among
    those, picks the one with the most APPROVED-stage indications. Falls back
    to highest-approval-count hit if no exact-name match exists, but logs a
    warning.
    """
    async with OpenTargetsClient(cache_dir=cache_dir) as client:
        data = await client._graphql(
            OPEN_TARGETS_BASE_URL, _OT_DRUG_SEARCH_QUERY, {"q": drug_name}
        )
        hits = [h for h in data["data"]["search"]["hits"] if h["entity"] == "drug"]
        if not hits:
            return None, []

        q_lower = drug_name.lower()
        # tuple shape: (name_matches_query, approved_count, chembl_id, name, approved_list)
        scored: list[tuple[bool, int, str, str, list[str]]] = []
        for h in hits:
            cid = h["id"]
            hit_name = (h.get("name") or "").lower()
            try:
                indications = await client.get_drug_indications(cid)
            except Exception as e:
                logger.warning(
                    "get_ot_approved_disease_mapping: get_drug_indications "
                    "failed for %s: %s", cid, e,
                )
                continue
            approved = sorted({
                (i.disease_name or "").lower()
                for i in indications
                if i.max_clinical_stage == "APPROVAL" and i.disease_name
            })
            name_matches = (
                hit_name == q_lower or hit_name.startswith(q_lower + " ")
            )
            scored.append((name_matches, len(approved), cid, hit_name, approved))

        if not scored:
            return None, []

        scored.sort(key=lambda r: (r[0], r[1]), reverse=True)
        name_match, _count, best_id, best_name, best_approved = scored[0]
        if not name_match:
            logger.warning(
                "get_ot_approved_disease_mapping: %r — no exact-name OT hit; "
                "falling back to highest-approval-count hit %s (%s)",
                drug_name, best_id, best_name,
            )
        return best_id, best_approved


async def get_ot_approved_disease_mapping(
    drug_name: str,
    candidate_diseases: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, bool]:
    """Decide, per candidate, whether the drug is FDA-approved for that disease,
    using Open Targets as the source of truth.

    Two-tier lookup:
      1. Curated short-circuit — exact, case-sensitive match against
         CURATED_FDA_APPROVED_CANDIDATES (force True) and
         CURATED_FDA_REJECTED_CANDIDATES (force False). Hits skip the OT
         fetch and the LLM call.
      2. LLM bridge — searches OT for the drug, fetches its APPROVED-stage
         indication list, batches the remaining candidates into one LLM call
         that decides yes/no per candidate via clinical-practice judgment
         (synonym/parent/lay-term).

    Args:
        drug_name: A single drug name (trade, generic/INN, or USAN).
        candidate_diseases: Disease names to check.
        cache_dir: Cache directory.

    Returns:
        Dict mapping each input candidate (verbatim) to True if the drug is
        FDA-approved for it, False otherwise. Every input candidate is always
        present as a key. Defaults to False on any failure (OT fetch, LLM
        parse).
    """
    result: dict[str, bool] = {c: False for c in candidate_diseases}

    if not drug_name or not candidate_diseases:
        return result

    # Curated short-circuit (force True / force False).
    approved_curated = set(CURATED_FDA_APPROVED_CANDIDATES.get(drug_name, []))
    rejected_curated = set(CURATED_FDA_REJECTED_CANDIDATES.get(drug_name, []))
    uncurated: list[str] = []
    for c in candidate_diseases:
        if c in approved_curated:
            result[c] = True
        elif c in rejected_curated:
            result[c] = False
        else:
            uncurated.append(c)

    if not uncurated:
        return result

    # OT lookup.
    try:
        chembl_id, ot_approved = await _ot_pick_drug_and_fetch_approved(
            drug_name, cache_dir
        )
    except Exception as e:
        logger.warning(
            "get_ot_approved_disease_mapping: OT fetch failed for %r: %s; "
            "returning False for %d uncurated candidate(s)",
            drug_name, e, len(uncurated),
        )
        return result

    logger.info(
        "get_ot_approved_disease_mapping: %r → chembl_id=%s, %d APPROVED indications",
        drug_name, chembl_id, len(ot_approved),
    )

    if not ot_approved:
        # Drug not found in OT, or OT has no APPROVED indications for it.
        # All uncurated candidates default to False.
        return result

    # LLM bridge.
    template = (_PROMPTS_DIR / "extract_ot_approval_bridge.txt").read_text()
    approved_list_text = "\n".join(f"  - {d}" for d in ot_approved)
    prompt = template.format(
        drug_name=drug_name,
        approved_list=approved_list_text,
        candidates_json=json.dumps(uncurated),
    )

    response = await query_small_llm(prompt)
    stripped = strip_markdown_fences(response)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        logger.error(
            "get_ot_approved_disease_mapping: failed to parse LLM response: %s",
            response,
        )
        return result

    if not isinstance(parsed, dict):
        logger.error(
            "get_ot_approved_disease_mapping: LLM returned non-dict: %s",
            type(parsed),
        )
        return result

    # Map LLM keys back to verbatim input candidates (case-insensitive),
    # scoped to uncurated candidates so a stray key cannot overwrite a
    # curated value.
    lower_to_verbatim = {c.lower(): c for c in uncurated}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        original = lower_to_verbatim.get(key.lower())
        if original is None:
            logger.warning(
                "get_ot_approved_disease_mapping: LLM returned unknown candidate %r, skipping",
                key,
            )
            continue
        if not isinstance(value, bool):
            logger.error(
                "get_ot_approved_disease_mapping: value for %r is not a bool: %s",
                key, type(value),
            )
            continue
        result[original] = value

    return result
