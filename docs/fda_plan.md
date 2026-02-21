# Plan: openFDA FAERS (Drug Adverse Event) Client

## Context

IndicationScout needs real-world post-market adverse event data for drug safety profiling. The openFDA FAERS API (`https://api.fda.gov/drug/event.json`) provides this — individual adverse event reports and aggregate reaction counts across millions of FDA safety reports. This complements the existing Open Targets FAERS summary (which only has name/count/LLR per event) with richer detail: seriousness breakdowns, individual report data, and frequency-ranked reaction profiles.

---

## Files to create

| # | File | Purpose |
|---|------|---------|
| 1 | `src/indication_scout/models/model_fda.py` | Pydantic models |
| 2 | `src/indication_scout/data_sources/fda.py` | `FDAClient(BaseClient)` |
| 3 | `tests/unit/test_fda.py` | Unit tests for parsers |
| 4 | `tests/integration/test_fda.py` | Integration tests against real API |

## Files to modify

| # | File | Change |
|---|------|--------|
| 5 | `src/indication_scout/constants.py` | Add `OPENFDA_BASE_URL`, `OPENFDA_MAX_LIMIT` |
| 6 | `src/indication_scout/config.py` | Add `openfda_api_key: str = ""` to Settings |
| 7 | `tests/integration/conftest.py` | Add `fda_client` fixture |

---

## Step 1 — Models (`src/indication_scout/models/model_fda.py`)

Two models, following `model_pubmed.py` pattern (plain BaseModel, no validators):

**`FAERSReactionCount`** — one row from the count endpoint (used by `get_top_reactions`)
- `term: str` — MedDRA preferred term (e.g. "NAUSEA")
- `count: int` — number of reports

**`FAERSEvent`** — one adverse event record (flat, one row per report, first drug/reaction picked)
- `medicinal_product: str` — drug name as reported (from `patient.drug[0].medicinalproduct`)
- `drug_indication: str | None = None` — what the drug was prescribed for (from `patient.drug[0].drugindication`)
- `reaction: str` — first MedDRA preferred term (from `patient.reaction[0].reactionmeddrapt`)
- `reaction_outcome: str | None = None` — numeric outcome code (from `patient.reaction[0].reactionoutcome`, e.g. "1"=recovered, "2"=recovering, "3"=not recovered, "4"=recovered with sequelae, "5"=fatal, "6"=unknown)
- `serious: str | None = None` — "1" serious, "2" not (top-level field)
- `company_numb: str | None = None` — manufacturer's report number (top-level field, e.g. "US-MERCK-1403USA005334")

Names are prefixed `FAERS*` to avoid collision with existing `AdverseEvent` in `model_open_targets.py`.

### Field source mapping (API response -> model)
| Model field | API JSON path |
|---|---|
| `medicinal_product` | `results[].patient.drug[0].medicinalproduct` |
| `drug_indication` | `results[].patient.drug[0].drugindication` |
| `reaction` | `results[].patient.reaction[0].reactionmeddrapt` |
| `reaction_outcome` | `results[].patient.reaction[0].reactionoutcome` |
| `serious` | `results[].serious` |
| `company_numb` | `results[].companynumb` |

## Step 2 — Constants (`src/indication_scout/constants.py`)

Add at bottom, following existing section-comment pattern:

```python
# -- openFDA ----------------------------------------------------------------
OPENFDA_BASE_URL: str = "https://api.fda.gov/drug/event.json"
OPENFDA_MAX_LIMIT: int = 1000
```

## Step 3 — Config (`src/indication_scout/config.py`)

Add under `# API Keys`:

```python
openfda_api_key: str = ""
```

Optional — openFDA works without a key (40 req/min), with key gets 240/min.

## Step 4 — Client (`src/indication_scout/data_sources/fda.py`)

Extends `BaseClient`. No `__init__` override needed (API key fetched via `get_settings()` per-call in `_build_params`).

`_source_name` returns `"openfda"`.

### Public methods

**`async get_top_reactions(drug_name: str, limit: int = 100) -> list[FAERSReactionCount]`**
- Searches `patient.drug.openfda.generic_name:"{drug_name}"`
- Uses `count=patient.reaction.reactionmeddrapt.exact` (`.exact` suffix required for proper MedDRA term aggregation)
- Returns list of `FAERSReactionCount` sorted by count descending
- Raises `DataSourceError` if no results (drug not found returns 404, handled by BaseClient)

**`async get_events(drug_name: str, limit: int = 100, serious_only: bool = False) -> list[FAERSEvent]`**
- Search mode, returns flat event records (first drug + first reaction per report)
- When `serious_only=True`, appends `AND serious:"1"` to search
- Returns empty list if no results

### Private methods

- `_parse_reaction_counts(data) -> list[FAERSReactionCount]` — parses count-mode JSON
- `_parse_events(data) -> list[FAERSEvent]` — parses search-mode JSON, flattens to one row per report
- `_build_params(*, search, count, limit, skip) -> dict` — builds query params, injects API key when configured

## Step 5 — Unit tests (`tests/unit/test_fda.py`)

Test the sync `_parse_*` methods with inline fixture dicts. No network calls.

- `test_parse_reaction_counts_returns_correct_fields` — two reactions, verify term + count
- `test_parse_reaction_counts_empty_results_raises_error` — `pytest.raises(DataSourceError)`
- `test_parse_reaction_counts_skips_incomplete_entries` — missing term/count filtered out
- `test_parse_events_returns_correct_fields` — full event, verify all 6 fields
- `test_parse_events_empty_results_returns_empty_list`
- `test_parse_events_missing_results_returns_empty_list`
- `test_parse_events_missing_optional_fields_are_none`
- `test_parse_events_picks_first_drug_and_reaction` — multiple drugs/reactions, assert first is picked
- `test_build_params_no_api_key` — patch `get_settings()`, assert no `api_key` in params
- `test_build_params_with_api_key` — patch `get_settings()`, assert `api_key` present

## Step 6 — Integration fixture (`tests/integration/conftest.py`)

Add `fda_client` fixture following existing pattern:

```python
@pytest.fixture
async def fda_client():
    c = FDAClient()
    yield c
    await c.close()
```

## Step 7 — Integration tests (`tests/integration/test_fda.py`)

Tests hit real openFDA API. Use a well-known drug (e.g. metformin or semaglutide).

- `test_get_top_reactions_returns_profile` — verify drug_name, reaction count, descending order
- `test_get_top_reactions_known_reaction_present` — verify a known common AE appears (e.g. NAUSEA for semaglutide)
- `test_get_events_returns_list` — verify event fields populated
- `test_get_events_serious_only_filter` — all returned events have `serious == "1"`
- `test_get_top_reactions_unknown_drug_raises_error` — nonexistent drug
- `test_get_events_unknown_drug_returns_empty` — nonexistent drug returns `[]`

---

## Implementation order

```
Steps 1-3 (models, constants, config) — independent, do first
Step 4 (client) — depends on 1-3
Steps 5-7 (tests) — depend on 4
```

## Verification

1. Run unit tests: `pytest tests/unit/test_fda.py -v`
2. Run linting: `ruff check src/indication_scout/models/model_fda.py src/indication_scout/data_sources/fda.py`
3. Run type checking: `mypy src/indication_scout/data_sources/fda.py src/indication_scout/models/model_fda.py`
4. Integration tests (manual, hits real API): `pytest tests/integration/test_fda.py -v`