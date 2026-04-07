"""Integration tests for the mechanism agent.

Hits real Anthropic, Open Targets, and ChEMBL APIs.
Expected values verified by a live run on 2026-04-06.
"""

import logging

from langchain_anthropic import ChatAnthropic

from indication_scout.agents.mechanism.mechanism_agent import (
    build_mechanism_agent,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Metformin + colorectal cancer
#
# Expected values verified by a live run on 2026-04-06.
# ------------------------------------------------------------------

# Diseases that must appear in the competitor map (stable approved indications)
_EXPECTED_COMPETITOR_DISEASES = {
    "type 2 diabetes mellitus",
    "obesity",
    "polycystic ovary syndrome",
    "non-alcoholic fatty liver disease",
    "cardiovascular disease",
    "colorectal cancer",
}

_EXPECTED_DRUG_INDICATIONS = {"diabetes mellitus", "type 2 diabetes mellitus"}

# Drug profile fields verified by live run
_EXPECTED_SYNONYMS_SUBSET = {"Metformin", "Metformina"}
_EXPECTED_TARGET_GENES_SUBSET = {"MT-ND1", "NDUFV1", "NDUFS2", "GPD2"}
_EXPECTED_MECHANISMS = {
    "Mitochondrial complex I (NADH dehydrogenase) inhibitor",
    "Mitochondrial glycerol-3-phosphate dehydrogenase inhibitor",
}
_EXPECTED_ATC_CODES = ["A10BA02"]
_EXPECTED_ATC_DESCRIPTIONS_SUBSET = {"Biguanides"}

# Diseases for which search queries must be generated
_EXPECTED_QUERY_DISEASES = {
    "cancer",
    "obesity",
    "non-alcoholic fatty liver disease",
    "cardiovascular disease",
}


async def test_metformin_mechanism_agent(test_cache_dir):
    """End-to-end: mechanism agent produces correct MechanismOutput for Metformin.

    Verifies:
    - known diseases appear in competitor map
    - drug_indications contains approved indications
    - drug_profile has correct name, synonyms, targets, mechanisms, ATC info
    - search_queries generated for key repurposing candidate diseases
    - narrative summary is non-empty
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    svc = RetrievalService(test_cache_dir)
    agent = build_mechanism_agent(llm, svc, date_before=None)

    output = await run_mechanism_agent(agent, "metformin", "colorectal cancer")

    assert isinstance(output, MechanismOutput)

    # --- competitors ---
    assert len(output.competitors) >= 10
    assert _EXPECTED_COMPETITOR_DISEASES.issubset(set(output.competitors.keys()))
    for disease, drugs in output.competitors.items():
        assert isinstance(drugs, list)
        assert len(drugs) >= 1

    # --- drug_indications ---
    assert _EXPECTED_DRUG_INDICATIONS.issubset(set(output.drug_indications))

    # --- drug_profile ---
    assert isinstance(output.drug_profile, DrugProfile)
    assert output.drug_profile.name == "METFORMIN"
    assert _EXPECTED_SYNONYMS_SUBSET.issubset(set(output.drug_profile.synonyms))
    assert len(output.drug_profile.synonyms) >= 3
    assert _EXPECTED_TARGET_GENES_SUBSET.issubset(set(output.drug_profile.target_gene_symbols))
    assert len(output.drug_profile.target_gene_symbols) >= 20
    assert _EXPECTED_MECHANISMS == set(output.drug_profile.mechanisms_of_action)
    assert output.drug_profile.atc_codes == _EXPECTED_ATC_CODES
    assert _EXPECTED_ATC_DESCRIPTIONS_SUBSET.issubset(set(output.drug_profile.atc_descriptions))
    assert output.drug_profile.drug_type == "Small molecule"

    # --- search_queries ---
    assert _EXPECTED_QUERY_DISEASES.issubset(set(output.search_queries.keys()))
    for disease, queries in output.search_queries.items():
        assert len(queries) >= 5, f"Expected >=5 queries for {disease!r}, got {len(queries)}"
        queries_lower = [q.lower() for q in queries]
        assert any("metformin" in q or "biguanide" in q for q in queries_lower), (
            f"No metformin/biguanide term in queries for {disease!r}"
        )

    # --- summary ---
    assert len(output.summary) > 200
