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

# -- PubMed / NCBI ----------------------------------------------------------
NCBI_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL: str = f"{NCBI_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL: str = f"{NCBI_BASE_URL}/efetch.fcgi"
PUBMED_SUMMARY_URL: str = f"{NCBI_BASE_URL}/esummary.fcgi"

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
STOP_KEYWORDS: dict[str, str] = {
    # Specific phrases first
    "lack of efficacy": "efficacy",
    "no benefit": "efficacy",
    "no significant difference": "efficacy",
    "side effect": "safety",
    "adverse event": "safety",
    "toxicity": "safety",
    "futility": "efficacy",
    # Then broader terms
    "efficacy": "efficacy",
    "enrollment": "enrollment",
    "accrual": "enrollment",
    "recruitment": "enrollment",
    "business": "business",
    "strategic": "business",
    "funding": "business",
    "commercial": "business",
    # "safety" and "adverse" are too broad without negation handling
    "safety concern": "safety",
    "safety signal": "safety",
    "clinical hold": "safety",
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

# -- Supervisor: mechanism-sourced candidate threshold -----------------------
# Minimum Open Targets overall_score for a mechanism-surfaced disease
# association to be promoted into the supervisor's investigation allowlist.
MECHANISM_ASSOCIATION_MIN_SCORE: float = 0.3
