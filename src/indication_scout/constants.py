"""Project-wide constants."""

from pathlib import Path

# -- Base client defaults ---------------------------------------------------
DEFAULT_TIMEOUT: float = 30.0
DEFAULT_MAX_RETRIES: int = 3

# -- Cache ------------------------------------------------------------------
DEFAULT_CACHE_DIR: Path = Path("_cache")
CACHE_TTL: int = 5 * 86400  # 5 days in seconds

# -- Open Targets -----------------------------------------------------------
OPEN_TARGETS_BASE_URL: str = "https://api.platform.opentargets.org/api/v4/graphql"

# -- Interaction type mapping (Open Targets) --------------------------------
INTERACTION_TYPE_MAP: dict[str, str] = {
    "intact": "physical",
    "signor": "signalling",
    "reactome": "enzymatic",
    "string": "functional",
}

# -- Stop-reason keywords → category (ClinicalTrials.gov) ------------------
STOP_KEYWORDS: dict[str, str] = {
    "efficacy": "efficacy",
    "futility": "efficacy",
    "lack of efficacy": "efficacy",
    "no benefit": "efficacy",
    "safety": "safety",
    "adverse": "safety",
    "toxicity": "safety",
    "side effect": "safety",
    "enrollment": "enrollment",
    "accrual": "enrollment",
    "recruitment": "enrollment",
    "business": "business",
    "strategic": "business",
    "funding": "business",
    "commercial": "business",
}

# -- openFDA ----------------------------------------------------------------
OPENFDA_BASE_URL: str = "https://api.fda.gov/drug/event.json"
OPENFDA_MAX_LIMIT: int = 1000
