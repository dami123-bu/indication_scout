"""FDA approval check service.

Uses openFDA drug labels + LLM extraction to identify which candidate
diseases are already FDA-approved for a given drug's trade names.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from indication_scout.constants import (
    CACHE_TTL,
    CURATED_FDA_APPROVED_CANDIDATES,
    CURATED_FDA_REJECTED_CANDIDATES,
    DEFAULT_CACHE_DIR,
)
from indication_scout.data_sources.chembl import (
    get_all_drug_names,
    resolve_drug_name,
)
from indication_scout.data_sources.fda import FDAClient
from indication_scout.services.llm import (
    parse_last_json_array,
    parse_last_json_object,
    query_llm,
    query_small_llm,
)
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_DRUG_APPROVAL_NS = "fda_drug_disease_approval"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _drug_approval_path(drug_name: str, cache_dir: Path) -> Path:
    """Return the per-drug cache file path for the approvals namespace.

    Slugifies the drug name to a filesystem-safe stem; appends an 8-char
    SHA suffix to disambiguate names that collapse to the same slug
    (e.g. "5-FU" vs "5_FU") and to handle empty slugs.
    """
    slug = _SLUG_RE.sub("_", drug_name.lower()).strip("_")
    suffix = hashlib.sha256(drug_name.encode()).hexdigest()[:8]
    stem = f"{slug}_{suffix}" if slug else suffix
    return cache_dir / _DRUG_APPROVAL_NS / f"{stem}.json"


def _load_drug_approvals(drug_name: str, cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the per-drug approvals file as {disease_lower: {verdict, cached_at, ttl}}.

    Returns an empty dict if the file is missing or unparseable. Expired
    entries are dropped lazily; the file is rewritten only when a caller
    invokes _save_drug_approvals.
    """
    path = _drug_approval_path(drug_name, cache_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return {}

    fresh: dict[str, dict[str, Any]] = {}
    now = datetime.now()
    for disease_key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        try:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            ttl = int(entry.get("ttl", CACHE_TTL))
            verdict = bool(entry["verdict"])
        except (KeyError, TypeError, ValueError):
            continue
        if (now - cached_at).total_seconds() > ttl:
            continue
        fresh[disease_key] = {
            "verdict": verdict,
            "cached_at": entry["cached_at"],
            "ttl": ttl,
        }
    return fresh


def _save_drug_approvals(
    drug_name: str,
    new_verdicts: dict[str, bool],
    cache_dir: Path,
    ttl: int = CACHE_TTL,
) -> None:
    """Merge new_verdicts into the per-drug file and write it back.

    Existing unexpired entries are preserved; new verdicts overwrite any
    prior entry for the same disease (refreshing its cached_at).
    """
    if not new_verdicts:
        return
    path = _drug_approval_path(drug_name, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_drug_approvals(drug_name, cache_dir)
    now_iso = datetime.now().isoformat()
    for disease, verdict in new_verdicts.items():
        existing[disease.lower()] = {
            "verdict": bool(verdict),
            "cached_at": now_iso,
            "ttl": ttl,
        }
    payload = {
        "ns": _DRUG_APPROVAL_NS,
        "drug_name": drug_name,
        "entries": existing,
    }
    path.write_text(json.dumps(payload, default=str, indent=2))

async def list_approved_indications_from_labels(
    label_texts: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[str]:
    """Extract the approved indications named in FDA label text, no candidate list.

    Companion to extract_approved_from_labels. That function asks "which of
    THESE candidates are approved per the label?" — useful for filtering a
    candidate list. This function asks "what indications does the label
    approve?" — useful for seeding the supervisor's drug-level briefing with
    approvals discovered up front, before any candidate list exists.

    Args:
        label_texts: Raw indications_and_usage strings from openFDA.
        cache_dir: Cache directory for storing LLM results.

    Returns:
        Deduplicated list of approved indication names extracted from the
        label text. Empty list if label_texts is empty, the LLM response is
        not parseable JSON, or the LLM returns a non-list. Order-preserving.
    """
    if not label_texts:
        return []

    cache_params = {"label_texts": sorted(label_texts)}
    cached = cache_get("fda_label_indications", cache_params, cache_dir)
    if cached is not None:
        return list(cached)

    template = (_PROMPTS_DIR / "list_label_indications.txt").read_text()
    prompt = template.format(label_texts="\n---\n".join(label_texts))

    response = await query_llm(prompt)
    parsed = parse_last_json_array(response)

    if parsed is None:
        logger.error(
            "list_approved_indications_from_labels: failed to parse LLM response: %s",
            response,
        )
        return []

    indications: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        indications.append(cleaned)

    cache_set(
        "fda_label_indications",
        cache_params,
        indications,
        cache_dir,
        ttl=CACHE_TTL,
    )
    return indications


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
    parsed = parse_last_json_array(response)

    if parsed is None:
        logger.error(
            "extract_approved_from_labels: failed to parse LLM response: %s", response
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
    # curated approved-candidate list (force True) and rejected-candidate
    # list (force False). Hits skip both the FDA fetch and the LLM call.
    approved_set = set(CURATED_FDA_APPROVED_CANDIDATES.get(drug_name, []))
    rejected_set = set(CURATED_FDA_REJECTED_CANDIDATES.get(drug_name, []))
    uncurated: list[str] = []
    for c in candidate_diseases:
        if c in approved_set:
            result[c] = True
        elif c in rejected_set:
            result[c] = False
        else:
            uncurated.append(c)

    if not uncurated:
        return result

    # Per-pair drug-disease verdict cache: one file per drug holding a
    # {disease_lower: {verdict, cached_at, ttl}} map. Apply unexpired hits
    # directly to result; only the still-missing candidates are sent to
    # ChEMBL/FDA/LLM below. New verdicts are merged back into the same file
    # after the LLM call. Curated entries are applied above and never cached
    # here, so curated overrides always win. TTL is per-verdict, preserving
    # the previous semantics where each (drug, disease) pair expired
    # independently.
    drug_cache = _load_drug_approvals(drug_name, cache_dir)
    still_missing: list[str] = []
    for c in uncurated:
        entry = drug_cache.get(c.lower())
        if entry is None:
            still_missing.append(c)
        else:
            result[c] = bool(entry["verdict"])

    if not still_missing:
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
            drug_name, len(drug_aliases), len(still_missing),
        )
        return result

    template = (_PROMPTS_DIR / "extract_fda_approval_single.txt").read_text()
    prompt = template.format(
        label_texts="\n---\n".join(label_texts),
        candidate_diseases=json.dumps(still_missing),
    )

    response = await query_llm(prompt)
    parsed = parse_last_json_object(response)

    if parsed is None:
        logger.error(
            "get_fda_approved_disease_mapping: failed to parse LLM response: %s",
            response,
        )
        return result

    # Map LLM keys back to verbatim input candidates (case-insensitive), scoped
    # to the still-missing candidates so a stray key cannot overwrite a curated
    # or already-cached value.
    lower_to_verbatim = {c.lower(): c for c in still_missing}
    llm_verdicts: dict[str, bool] = {}
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
        llm_verdicts[original] = value

    # Merge fresh verdicts into the per-drug cache file. Only candidates the
    # LLM actually returned a bool for are cached — parse failures or skipped
    # candidates remain uncached so a future call can retry them.
    _save_drug_approvals(drug_name, llm_verdicts, cache_dir, ttl=CACHE_TTL)

    return result


