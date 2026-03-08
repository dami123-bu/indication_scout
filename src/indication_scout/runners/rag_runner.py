"""End-to-end RAG pipeline runner for drug repurposing evidence retrieval."""

import asyncio
import logging
import time

from sqlalchemy.orm import Session

from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.db.session import get_db
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import (
    build_drug_profile,
    expand_search_terms,
    fetch_and_cache,
    semantic_search,
    synthesize,
)

logger = logging.getLogger(__name__)


async def run_rag(drug_name: str, db: Session) -> dict[str, EvidenceSummary]:
    """Run the full RAG pipeline for a drug across its top competitor indications.

    Steps for each disease indication:
      1. get_drug_competitors  — fetch top 10 disease indications from Open Targets
      2. build_drug_profile    — fetch drug/target data from Open Targets + ChEMBL
      3. expand_search_terms   — LLM generates diverse PubMed queries
      4. fetch_and_cache       — search PubMed, embed new abstracts, store in pgvector
      5. semantic_search       — cosine similarity search for top-k abstracts
      6. synthesize            — LLM synthesizes evidence into EvidenceSummary

    Args:
        drug_name: Common drug name (e.g. "metformin").
        db: Active SQLAlchemy session connected to pgvector DB.

    Returns:
        Dict mapping disease name to EvidenceSummary for each indication.
    """
    logger.info("run_rag: starting pipeline for drug '%s'", drug_name)

    async with OpenTargetsClient() as open_targets_client:
        top_15 = await open_targets_client.get_drug_competitors(drug_name)

    logger.info(
        "run_rag: %d disease indications found for \n%s",
        len(top_15),
        ", ".join(top_15.keys()),
    )

    drug_profile = await build_drug_profile(drug_name)
    logger.info(
        "run_rag: drug profile built — %d targets",
        len(drug_profile.target_gene_symbols),
    )

    results: dict[str, EvidenceSummary] = {}

    for disease in top_15:
        logger.info("run_rag: processing disease '%s'", disease)

        t0 = time.perf_counter()
        queries = await expand_search_terms(drug_name, disease, drug_profile)
        logger.info(
            "run_rag: expand_search_terms took %.2fs, %d queries for '%s'",
            time.perf_counter() - t0,
            len(queries),
            disease,
        )

        t0 = time.perf_counter()
        pmids = await fetch_and_cache(queries, db)
        logger.info(
            "run_rag: fetch_and_cache took %.2fs, %d PMIDs for '%s'",
            time.perf_counter() - t0,
            len(pmids),
            disease,
        )

        t0 = time.perf_counter()
        top_abstracts = await semantic_search(disease, drug_name, pmids, db)
        logger.info(
            "run_rag: semantic_search took %.2fs, %d abstracts for '%s'",
            time.perf_counter() - t0,
            len(top_abstracts),
            disease,
        )

        t0 = time.perf_counter()
        evidence = await synthesize(drug_name, disease, top_abstracts)
        logger.info(
            "run_rag: synthesize took %.2fs, strength=%s for '%s'",
            time.perf_counter() - t0,
            evidence.strength,
            disease,
        )

        results[disease] = evidence

        print(f"  Strength: {evidence.strength}")
        print(f"  Adverse: {evidence.has_adverse_effects}")
        print(f"  Summary: {evidence.summary[:100]}...")

    ranking = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}
    sorted_results = sorted(
        results.items(), key=lambda x: ranking[x[1].strength], reverse=True
    )

    print(f"\n=== Final Ranking for {drug_name} ===")
    for disease, summary in sorted_results:
        flag = "⚠️" if summary.has_adverse_effects else "✓"
        print(f"  {flag} {disease}: {summary.strength}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _main() -> None:
        db = next(get_db())
        try:
            results = await run_rag("empagliflozin", db)
            for disease, evidence in results.items():
                print(f"\n=== {disease} ===")
                print(evidence.model_dump_json(indent=2))
        finally:
            db.close()

    asyncio.run(_main())
