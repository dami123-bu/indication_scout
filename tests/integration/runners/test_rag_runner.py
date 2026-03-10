"""Integration tests for runners/rag_runner."""

import logging

from indication_scout.runners.rag_runner import run_rag

logger = logging.getLogger(__name__)

# Disease keys and EvidenceSummary values verified from a live run on 2026-03-08
# using empagliflozin against its top 15 disease indications from Open Targets.
# anchor_pmids: landmark papers that must always appear in supporting_pmids.
# study_count and study_types are not asserted — they vary with LLM output.
_EXPECTED = {
    "diabetic nephropathy": {
        "strength": "strong",
        "has_adverse_effects": False,
        "anchor_pmids": ["27817207", "34752913"],
    },
    "non-alcoholic fatty liver disease": {
        "strength": "strong",
        "has_adverse_effects": False,
        "anchor_pmids": ["32975679", "33586120"],
    },
    "atrial fibrillation": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "dilated cardiomyopathy": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "insulin resistance": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "myocardial infarction": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": ["26488032"],
    },
    "metabolic syndrome": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "polycystic ovary syndrome": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "diabetic retinopathy": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "coronary artery disease": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": ["26488032"],
    },
    "type 1 diabetes mellitus": {
        "strength": "moderate",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "nasopharyngeal neoplasm": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "hypertrophic cardiomyopathy": {
        "strength": "weak",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "gestational diabetes": {
        "strength": "weak",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "severe acute respiratory syndrome": {
        "strength": "none",
        "has_adverse_effects": False,
        "anchor_pmids": ["37865101"],
    },
}


# Disease keys and EvidenceSummary values verified from a live run on 2026-03-07
# using semaglutide against its top 15 disease indications from Open Targets.
# anchor_pmids: landmark papers that must always appear in supporting_pmids.
# study_count and study_types are not asserted — they vary with LLM output.
# anchor_pmids will be populated after a confirmed stable semaglutide run.
_EXPECTED_SEMAGLUTIDE = {
    "coronary artery disease": {
        "strength": "strong",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "cardiovascular disease": {
        "strength": "strong",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "diabetic nephropathy": {
        "strength": "strong",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "abnormal glucose tolerance": {
        "strength": "strong",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "acute myocardial infarction": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "non-alcoholic steatohepatitis": {
        "strength": "strong",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "type 1 diabetes mellitus": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "nutritional disorder": {
        "strength": "moderate",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "metabolic syndrome": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "polycystic ovary syndrome": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "pseudotumor cerebri": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "acute coronary syndrome": {
        "strength": "moderate",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
    "eating disorder": {
        "strength": "weak",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "major depressive disorder": {
        "strength": "weak",
        "has_adverse_effects": True,
        "anchor_pmids": [],
    },
    "delirium": {
        "strength": "none",
        "has_adverse_effects": False,
        "anchor_pmids": [],
    },
}


def _assert_expected_diseases(results: dict, expected: dict) -> None:
    """Assert EvidenceSummary fields for each disease present in expected.

    Only diseases in the expected dict are checked — diseases returned by the
    live run that are not in expected are silently ignored. This tolerates
    Open Targets returning a different top-15 across runs due to cache drift.

    For each disease:
    - strength and has_adverse_effects are asserted exactly.
    - anchor_pmids (landmark papers) are asserted as a required subset of supporting_pmids.
    - supporting_pmids is asserted non-empty for any disease with strength != "none".
    - study_count and study_types are not asserted — they vary with LLM output.
    """
    for disease, exp in expected.items():
        assert disease in results, f"expected disease '{disease}' not in results"
        evidence = results[disease]
        assert evidence.strength == exp["strength"], f"{disease}: strength mismatch"
        assert evidence.has_adverse_effects == exp["has_adverse_effects"], \
            f"{disease}: has_adverse_effects mismatch (got {evidence.has_adverse_effects}, expected {exp['has_adverse_effects']})"
        anchor_pmids = set(exp.get("anchor_pmids", []))
        if anchor_pmids:
            assert anchor_pmids <= set(evidence.supporting_pmids), (
                f"{disease}: anchor PMIDs missing from supporting_pmids: "
                f"{anchor_pmids - set(evidence.supporting_pmids)}"
            )
        if exp["strength"] != "none":
            assert len(evidence.supporting_pmids) > 0, f"{disease}: supporting_pmids is empty"
        assert len(evidence.summary) > 0, f"{disease}: summary is empty"
        assert len(evidence.key_findings) > 0, f"{disease}: key_findings is empty"


async def test_run_rag_empagliflozin(db_session_truncating, test_cache_dir):
    """run_rag returns correct EvidenceSummary fields for known empagliflozin disease indications.

    Only diseases present in _EXPECTED are checked — the live top-15 from Open
    Targets may vary between runs as per-disease caches expire.
    """
    results = await run_rag("empagliflozin", db_session_truncating, cache_dir=test_cache_dir)
    for disease, evidence in results.items():
        logger.info(
            "empagliflozin | %s | strength=%s adverse=%s pmids=%s",
            disease,
            evidence.strength,
            evidence.has_adverse_effects,
            evidence.supporting_pmids,
        )
    _assert_expected_diseases(results, _EXPECTED)


async def test_run_rag_semaglutide(db_session_truncating, test_cache_dir):
    """run_rag returns correct EvidenceSummary fields for known semaglutide disease indications.

    Only diseases present in _EXPECTED_SEMAGLUTIDE are checked — the live top-15
    from Open Targets may vary between runs as per-disease caches expire.
    """
    results = await run_rag("semaglutide", db_session_truncating, cache_dir=test_cache_dir)
    _assert_expected_diseases(results, _EXPECTED_SEMAGLUTIDE)
