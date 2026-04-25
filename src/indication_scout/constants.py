"""Project-wide constants.

Tunable numeric limits (timeouts, retries, top-k, max results, concurrency,
etc.) live in config.py Settings and are overridable via environment variables
or alternate .env files (ENV_FILE=.env.test).

This file holds values that are structurally fixed: URLs, lookup maps,
keyword lists, and directory paths.
"""

from pathlib import Path

# -- LLM defaults -----------------------------------------------------------
DEFAULT_LLM_MODEL: str = "claude-sonnet-4-6"

# -- Cache ------------------------------------------------------------------
# Anchored to the project root (two levels above this package's src/ dir) so
# that a single _cache/ directory is used regardless of the working directory
# from which tests or scripts are launched.
_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
DEFAULT_CACHE_DIR: Path = _PROJECT_ROOT / "_cache"
TEST_CACHE_DIR: Path = _PROJECT_ROOT / "_cache_test"
CACHE_TTL: int = 5 * 86400  # 5 days in seconds

# -- Open Targets -----------------------------------------------------------
OPEN_TARGETS_BASE_URL: str = "https://api.platform.opentargets.org/api/v4/graphql"

# -- ChEMBL -----------------------------------------------------------------
CHEMBL_BASE_URL: str = "https://www.ebi.ac.uk/chembl/api/data"

# -- ClinicalTrials.gov -----------------------------------------------------
CLINICAL_TRIALS_BASE_URL: str = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER: str = "(PHASE2 OR PHASE3 OR PHASE4)"
CLINICAL_TRIALS_RECENT_START_YEAR: str = "2024"
# Safety ceilings for MeSH-filtered paths. CT.gov pages are 100 studies each.
# _count_trials walks the unfiltered result set to count post-MeSH survivors;
# cap at 10 pages (≈1000 studies) to avoid unbounded walks on broad indications.
CLINICAL_TRIALS_COUNT_PAGE_CAP: int = 10
# get_landscape fetches unbounded when MeSH-filtering, then caps post-filter to
# max_results. Cap the pre-filter fetch at 20 pages (≈2000 studies) to bound
# latency while giving room for narrow MeSH subsets of broad Essie results.
CLINICAL_TRIALS_LANDSCAPE_FETCH_CAP: int = 20

# -- PubMed / NCBI ----------------------------------------------------------
NCBI_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL: str = f"{NCBI_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL: str = f"{NCBI_BASE_URL}/efetch.fcgi"
PUBMED_SUMMARY_URL: str = f"{NCBI_BASE_URL}/esummary.fcgi"

# -- MeSH resolver ----------------------------------------------------------
NCBI_ESEARCH_URL: str = f"{NCBI_BASE_URL}/esearch.fcgi"
NCBI_ESUMMARY_URL: str = f"{NCBI_BASE_URL}/esummary.fcgi"
MESH_RESOLVER_TTL_SECONDS: int = 60 * 60 * 24 * 30  # 30 days

# -- Clinical stage ranking (Open Targets) ----------------------------------
# Maps maximumClinicalStage / maxClinicalStage string values to numeric ranks
# for comparison. Higher rank = further in pipeline.
CLINICAL_STAGE_RANK: dict[str, int] = {
    "UNKNOWN": 0,
    "PRECLINICAL": 1,
    "IND": 2,
    "EARLY_PHASE_1": 3,
    "PHASE_1": 4,
    "PHASE_1_2": 5,
    "PHASE_2": 6,
    "PHASE_2_3": 7,
    "PHASE_3": 8,
    "PREAPPROVAL": 9,
    "APPROVAL": 10,
}

# -- Interaction type mapping (Open Targets) --------------------------------
INTERACTION_TYPE_MAP: dict[str, str] = {
    "intact": "physical",
    "signor": "signalling",
    "reactome": "enzymatic",
    "string": "functional",
}

# -- Overly broad disease terms (used for filtering in competitors and normalization) --
BROADENING_BLOCKLIST: frozenset[str] = frozenset(
    {
        "cancer",
        "carcinoma",
        "tumor",
        "tumour",
        "neoplasm",
        "malignancy",
        "disease",
        "disorder",
        "syndrome",
        "indication",
        "pain",
    }
)

# -- Stop-reason keywords → category (ClinicalTrials.gov) ------------------
# Order matters: _classify_stop_reason takes the first match, so more specific
# phrases must appear before broader ones. Examples observed in real CT.gov
# stop-reason text that motivated each entry are noted inline.
STOP_KEYWORDS: dict[str, str] = {
    # -- Disambiguate "futility" variants (specific → general) ---------------
    # "enrollment futility" = operational futility, not efficacy
    "enrollment futility": "enrollment",
    "recruitment futility": "enrollment",
    # otherwise "futility" is an efficacy stop
    "futility": "efficacy",

    # -- Efficacy: explicit-phrase forms ------------------------------------
    "lack of efficacy": "efficacy",
    "no benefit": "efficacy",
    "no significant difference": "efficacy",
    # seen on solanezumab Phase 3s; common phrasing for endpoint misses
    "did not meet the primary endpoint": "efficacy",
    "did not meet the study's primary endpoint": "efficacy",
    "did not meet the target efficacy": "efficacy",
    "primary endpoint was not met": "efficacy",
    "primary endpoint not met": "efficacy",
    "not meet its primary endpoint": "efficacy",
    # seen on metformin × diabetes (NCT02111096) and similar benefit/risk phrasing
    "benefit-risk profile did not support": "efficacy",
    "not support continued development": "efficacy",
    "insufficient scientific evidence": "efficacy",
    "meaningful benefit": "efficacy",
    "insufficient target engagement": "efficacy",

    # -- Safety: explicit-phrase forms (hepatic is the dominant real-world form)
    "liver safety": "safety",
    "hepatic safety": "safety",
    "liver enzyme": "safety",  # covers "liver enzymes", "liver enzyme elevations"
    "transaminase": "safety",  # covers "transaminases", "transaminase elevations"
    "hepatotoxicity": "safety",
    "nonclinical safety": "safety",
    "safety finding": "safety",  # covers "safety findings"
    # seen on atabecestat × AD (NCT02569398) — canonical hepatotoxicity stop
    "change in benefit-risk profile": "safety",
    "side effect": "safety",
    "adverse event": "safety",
    "toxicity": "safety",
    "safety concern": "safety",
    "safety signal": "safety",
    "clinical hold": "safety",

    # -- Broader terms (these fire if nothing above matched) -----------------
    "efficacy": "efficacy",
    "enrollment": "enrollment",
    "accrual": "enrollment",
    "recruitment": "enrollment",
    "business": "business",
    "strategic": "business",
    "funding": "business",
    "commercial": "business",
}

NEGATION_PREFIXES: list[str] = ["no ", "not ", "unrelated to ", "without ", "non-"]

# -- Vaccine name keywords (for landscape competitor filtering) -------------
# Biologicals whose names match any of these substrings are classified as
# vaccines and excluded from the competitive landscape — they are not
# mechanism competitors for small-molecule or biologic drugs.
VACCINE_NAME_KEYWORDS: frozenset[str] = frozenset(
    {
        "vaccine",
        "vax",
        "immuniz",
        "immunis",
        "toxoid",
        "vacuna",
    }
)

MECHANISM_SIGNAL_KEYS: frozenset[str] = frozenset({"genetic_association", "literature", "affected_pathway"})

# Drug action types that produce loss-of-function on the target.
LOF_ACTION_TYPES: frozenset[str] = frozenset({
    "INHIBITOR",
    "ANTAGONIST",
    "NEGATIVE ALLOSTERIC MODULATOR",
    "NEGATIVE MODULATOR",
    "BLOCKER",
})

# Drug action types that produce gain-of-function on the target.
GOF_ACTION_TYPES: frozenset[str] = frozenset({
    "AGONIST",
    "ACTIVATOR",
    "POSITIVE ALLOSTERIC MODULATOR",
    "POSITIVE MODULATOR",
    "PARTIAL AGONIST",
})

# Number of top POSITIVE repurposing candidates the mechanism agent surfaces.
MECHANISM_TOP_CANDIDATES: int = 5

# -- openFDA ----------------------------------------------------------------
OPENFDA_BASE_URL: str = "https://api.fda.gov/drug/label.json"
OPENFDA_LABEL_LIMIT: int = 5

# Curated per-drug list of candidate disease phrasings to short-circuit as
# FDA-approved (return True without calling the LLM). Acts strictly as an
# LLM backstop: only add candidate phrasings the LLM-against-label flow
# misses (narrower subsets, lay synonyms, unusual phrasings, label coverage
# gaps, etc.) for a given drug. When a candidate disease string EXACTLY
# matches (case-sensitive, no normalization) any string in the drug's list,
# the approval check short-circuits to True without calling the LLM.
# Candidates not in the list fall through to the LLM-against-label flow.
CURATED_FDA_APPROVED_CANDIDATES: dict[str, list[str]] = {
    "semaglutide": [
        "morbid obesity",
        # FLOW indication (CKD in T2D); the label phrases CKD as both target
        # and qualifier population, which the LLM reads as ambiguous → no.
        "chronic kidney disease",
    ],
    "atorvastatin": [
        "hypercholesterolemia",
        "high cholesterol",
        "coronary heart disease",
        "coronary artery disease",
    ],
    "imatinib": ["chronic myeloid leukemia"],
    # openFDA top-N label fetch misses these real approvals (e.g. PMDD lives
    # on a Sarafem-era label that doesn't surface in fluoxetine's top labels;
    # bipolar I lives on the Symbyax combination label).
    "fluoxetine": ["premenstrual dysphoric disorder", "bipolar disorder"],
    "sarafem": ["premenstrual dysphoric disorder"],
    "levothyroxine": ["goiter", "tsh suppression"],
    "amoxicillin": ["pneumonia"],
    # Migraine is on branded ibuprofen labels (Advil Migraine, Motrin Migraine)
    # but typically not on the generic ibuprofen label that openFDA serves.
    "ibuprofen": ["migraine disorder"],
    # OT mis-tags real FDA approvals as PHASE_3 (their APPROVAL flag lags
    # the label in some cases). These are all on actual FDA labels.
    "tofacitinib": [
        "ulcerative colitis",
        "juvenile idiopathic arthritis",
        "ankylosing spondylitis",
        "psoriatic arthritis",
    ],
    "ciprofloxacin": ["anthrax", "plague"],
    "omeprazole": [
        "helicobacter pylori infection",
        "zollinger-ellison syndrome",
    ],
    # Lay disease names that OT records under syndrome/organism phrasings
    # (e.g. urethritis caused by C. trachomatis) — bridge can't connect.
    "azithromycin": ["chlamydia"],
    "doxycycline": ["chlamydia", "lyme disease"],
    # Both openFDA snapshot and OT miss these real Flagyl approvals.
    "metronidazole": ["giardiasis", "clostridium difficile infection"],
    # Symbyax (olanzapine + fluoxetine) carries the TRD approval; OT lists
    # MDD on olanzapine alone, which conflates the combo's approval.
    "olanzapine": ["treatment-resistant depression"],
}


# Curated per-drug list of candidate disease phrasings to short-circuit as
# NOT FDA-approved (return False without calling the LLM). Mirrors
# CURATED_FDA_APPROVED_CANDIDATES but for the rejection direction. Use
# sparingly: only when the LLM bridge over-matches on narrow-population
# approvals or symptom-specific approvals where the bare disease name
# would be misleading. When a candidate string EXACTLY matches (case-
# sensitive, no normalization) any string in the drug's list, the approval
# check short-circuits to False without calling the LLM.
CURATED_FDA_REJECTED_CANDIDATES: dict[str, list[str]] = {
    # Approved for "irritability associated with autistic disorder" only —
    # bare "autism" misleadingly suggests the disease is treated broadly.
    "risperidone": ["autism"],
    # Tamsulosin treats BPH (benign prostatic hyperplasia), NOT prostate
    # cancer; OT may surface unrelated prostate-related approvals.
    "tamsulosin": ["prostate cancer"],
    # RLS is approved for gabapentin enacarbil (Horizant), a different drug
    # entity from gabapentin proper. The LLM conflates them.
    "gabapentin": ["restless legs syndrome"],
    # MDD as monotherapy is NOT approved for olanzapine; only TRD as the
    # Symbyax combination with fluoxetine.
    "olanzapine": ["major depressive disorder"],
}

# -- Supervisor: mechanism-sourced candidate threshold -----------------------
# Minimum Open Targets overall_score for a mechanism-surfaced disease
# association to be promoted into the supervisor's investigation allowlist.
MECHANISM_ASSOCIATION_MIN_SCORE: float = 0.3
