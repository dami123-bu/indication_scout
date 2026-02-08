# Architecture

## BaseClient Infrastructure

All data source clients inherit from `BaseClient`, which provides common infrastructure for reliable API communication.

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BaseClient                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Request Methods                                                             │
│  ├── _request()   - Low-level HTTP with retry + rate limiting               │
│  ├── _graphql()   - GraphQL queries with caching                            │
│  └── _rest_get()  - REST GET requests with caching                          │
│                                                                              │
│  Rate Limiting (TokenBucketRateLimiter)                                      │
│  └── Token bucket algorithm, configurable requests/second and burst         │
│                                                                              │
│  Retry Logic                                                                 │
│  └── Exponential backoff with jitter, configurable max retries              │
│                                                                              │
│  Disk Cache (DiskCache)                                                      │
│  └── JSON files in _cache/, keyed by namespace + params hash, TTL-based     │
│                                                                              │
│  Response Wrapper (PartialResult)                                            │
│  └── Graceful degradation: is_complete, data, errors                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```python
ClientConfig(
    timeout_seconds=30.0,
    retry=RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
    ),
    rate_limit=RateLimitConfig(
        requests_per_second=5.0,
        burst=10,
    ),
    cache=CacheConfig(
        enabled=True,
        directory=Path("_cache"),
    ),
)
```

### DiskCache

- Location: `_cache/` relative to working directory
- File naming: `{namespace}_{hash}.json` where hash is MD5 of sorted params
- Each file contains: `{"timestamp": epoch, "data": ...}`
- TTL checked on read; expired entries return cache miss

### PartialResult

Wrapper for API responses enabling graceful degradation:

```python
PartialResult(
    is_complete=True,   # False if errors occurred
    data={...},         # Response data (may be partial)
    errors=[...],       # List of error messages
)
```

Clients check `result.is_complete` before processing; partial data can still be used when appropriate.

---

## Open Targets Data Structure

The `OpenTargetsClient` provides two independent data objects: `DrugData` and `TargetData`. These are fetched separately via `get_drug()` and `get_target_data()` methods, each with their own cache.

### DrugData

```
DrugData
 |-- chembl_id: str
 |-- name: str
 |-- synonyms: list[str]
 |-- trade_names: list[str]
 |-- drug_type: str
 |-- is_approved: bool
 |-- max_clinical_phase: float
 |-- year_first_approved: int | None
 |-- warnings: list[DrugWarning]
 |        |-- warning_type: str
 |        |-- description: str | None
 |        |-- toxicity_class: str | None
 |        |-- country: str | None
 |        |-- year: int | None
 |        +-- efo_id: str | None
 |-- indications: list[Indication]
 |        |-- disease_id: str
 |        |-- disease_name: str
 |        |-- max_phase: float
 |        +-- references: list[dict]
 |-- targets: list[DrugTarget]
 |        |-- target_id: str  ─────────────────> use with get_target_data()
 |        |-- target_symbol: str
 |        |-- mechanism_of_action: str
 |        +-- action_type: str | None
 |-- adverse_events: list[AdverseEvent]
 |        |-- name: str
 |        |-- meddra_code: str | None
 |        |-- count: int
 |        +-- log_likelihood_ratio: float
 +-- adverse_events_critical_value: float
```

### TargetData

```
TargetData
 |-- target_id: str
 |-- symbol: str
 |-- name: str
 |-- associations: list[Association]
 |        |-- disease_id: str
 |        |-- disease_name: str
 |        |-- overall_score: float
 |        |-- datatype_scores: dict[str, float]
 |        +-- therapeutic_areas: list[str]
 |-- pathways: list[Pathway]
 |        |-- pathway_id: str
 |        |-- pathway_name: str
 |        +-- top_level_pathway: str
 |-- interactions: list[Interaction]
 |        |-- interacting_target_id: str
 |        |-- interacting_target_symbol: str
 |        |-- interaction_score: float
 |        |-- source_database: str
 |        |-- biological_role: str
 |        |-- evidence_count: int
 |        +-- interaction_type: str | None
 |-- drug_summaries: list[DrugSummary]
 |        |-- drug_id: str
 |        |-- drug_name: str
 |        |-- disease_id: str
 |        |-- disease_name: str
 |        |-- phase: float
 |        |-- status: str | None
 |        |-- mechanism_of_action: str
 |        +-- clinical_trial_ids: list[str]
 |-- expressions: list[TissueExpression]
 |        |-- tissue_id: str
 |        |-- tissue_name: str
 |        |-- tissue_anatomical_system: str
 |        |-- rna: RNAExpression
 |        |        |-- value: float
 |        |        |-- quantile: int
 |        |        +-- unit: str
 |        +-- protein: ProteinExpression
 |                 |-- level: int
 |                 |-- reliability: bool
 |                 +-- cell_types: list[CellTypeExpression]
 |                          |-- name: str
 |                          |-- level: int
 |                          +-- reliability: bool
 |-- mouse_phenotypes: list[MousePhenotype]
 |        |-- phenotype_id: str
 |        |-- phenotype_label: str
 |        |-- phenotype_categories: list[str]
 |        +-- biological_models: list[BiologicalModel]
 |                 |-- allelic_composition: str
 |                 |-- genetic_background: str
 |                 |-- literature: list[str]
 |                 +-- model_id: str
 |-- safety_liabilities: list[SafetyLiability]
 |        |-- event: str | None
 |        |-- event_id: str | None
 |        |-- effects: list[SafetyEffect]
 |        |        |-- direction: str
 |        |        +-- dosing: str | None
 |        |-- datasource: str | None
 |        |-- literature: str | None
 |        +-- url: str | None
 +-- genetic_constraint: list[GeneticConstraint]
          |-- constraint_type: str
          |-- oe: float | None
          |-- oe_lower: float | None
          |-- oe_upper: float | None
          |-- score: float | None
          +-- upper_bin: int | None
```

**Key design**: `DrugTarget` (inside `DrugData`) only holds lightweight reference data. To get the full target data, call `get_target_data(target_id)` separately. This allows targets to be cached independently and shared across drugs.

---

## Helper Properties on DrugData

```python
drug.approved_disease_ids      # set[str] - Disease IDs with phase 4+
drug.investigated_disease_ids  # set[str] - All disease IDs being pursued
```

---

## Three Disease Links

| Path | What it answers |
|------|-----------------|
| `drug.indications` | What diseases is **this drug** being tested for? |
| `target.associations` | What diseases is **this target** linked to (by any evidence)? |
| `target.drug_summaries` | What **other drugs** target this protein, and for what diseases? |

---

## Finding Whitespace Indications

The system finds repurposing opportunities by:

1. Get `target.associations` - diseases linked to the target
2. Filter out `drug.indications` - diseases already being pursued
3. What remains = potential new indications

Example: If GLP1R has a high association score with NASH, but semaglutide's `indications` list doesn't include NASH, that's a whitespace opportunity.

---

## OpenTargetsClient Implementation

### Data Flow

```
                          ┌─────────────────────────────────────────────────┐
                          │             OpenTargetsClient                   │
                          │                                                 │
  get_drug("semaglutide") │   1. _resolve_drug_name() ──> SEARCH_QUERY     │
           │              │          └─> returns ChEMBL ID                  │
           ▼              │                                                 │
   ┌───────────────┐      │   2. Check _drug_cache (in-memory)             │
   │  Drug Name    │──────│          └─> if hit, return cached DrugData    │
   │  "semaglutide"│      │                                                 │
   └───────────────┘      │   3. Check disk cache (DiskCache)              │
                          │          └─> if hit, deserialize & return      │
                          │                                                 │
                          │   4. _fetch_drug() ──> DRUG_QUERY              │
                          │          └─> parse response ──> DrugData       │
                          │          └─> store in both caches              │
                          └─────────────────────────────────────────────────┘

get_target_data("ENSG...") follows the same pattern with TARGET_QUERY
```

### Two-Level Caching

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Cache Hierarchy                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Level 1: In-Memory Cache (per client instance)                     │
│  ├── _drug_cache: dict[str, DrugData]      # keyed by chembl_id     │
│  └── _target_data_cache: dict[str, TargetData]  # keyed by target_id│
│                                                                      │
│  Level 2: Disk Cache (shared across sessions)                       │
│  ├── namespace: "drug"    + params: {"chembl_id": ...}              │
│  └── namespace: "target"  + params: {"target_id": ...}              │
│                                                                      │
│  TTL: 5 days (CACHE_TTL = 5 * 86400)                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### GraphQL Queries

The client uses four GraphQL queries against `https://api.platform.opentargets.org/api/v4/graphql`:

| Query | Purpose | Variables |
|-------|---------|-----------|
| `SEARCH_QUERY` | Resolve drug name to ChEMBL ID | `q: str` |
| `DRUG_QUERY` | Fetch full drug data | `id: str` (ChEMBL ID) |
| `TARGET_QUERY` | Fetch full target data | `id: str` (Ensembl ID) |
| `ASSOCIATIONS_PAGE_QUERY` | Paginate associations (if > 500) | `id, index, size` |

### Parsing Pipeline

Raw GraphQL responses are parsed into Pydantic models via dedicated parser methods:

```
GraphQL Response                    Parser Method                  Pydantic Model
─────────────────                   ─────────────                  ──────────────
drug { ... }                   ──>  _parse_drug_data()        ──>  DrugData
  └── mechanismsOfAction       ──>    (inline)                ──>    DrugTarget
  └── indications              ──>    (inline)                ──>    Indication
  └── drugWarnings             ──>    (inline)                ──>    DrugWarning
  └── adverseEvents            ──>  _parse_adverse_event()    ──>    AdverseEvent

target { ... }                 ──>  _parse_target_data()      ──>  TargetData
  └── associatedDiseases       ──>  _parse_association()      ──>    Association
  └── pathways                 ──>  _parse_pathway()          ──>    Pathway
  └── interactions             ──>  _parse_interaction()      ──>    Interaction
  └── knownDrugs               ──>  _parse_drug_summary()     ──>    DrugSummary
  └── expressions              ──>  _parse_expression()       ──>    TissueExpression
  └── mousePhenotypes          ──>  _parse_phenotype()        ──>    MousePhenotype
  └── safetyLiabilities        ──>  _parse_safety_liability() ──>    SafetyLiability
  └── geneticConstraint        ──>  _parse_constraint()       ──>    GeneticConstraint
```

### Interaction Type Mapping

The `interaction_type` field is derived from `source_database` using `INTERACTION_TYPE_MAP`:

| source_database | interaction_type |
|-----------------|------------------|
| `intact` | `physical` |
| `string` | `functional` |
| `signor` | `signalling` |
| `reactome` | `enzymatic` |

### Pagination

Associations are paginated when count exceeds 500:

```python
# In _fetch_target():
if len(target_data.associations) >= 500:
    target_data.associations = await self._paginate_associations(target_id)

# _paginate_associations() loops through pages of 500 until all fetched
```

---

## Client API

```python
client = OpenTargetsClient()

# Fetch drug data (cached)
drug = await client.get_drug("semaglutide")

# Fetch target data separately (cached independently)
target = await client.get_target_data("ENSG00000112164")

# Accessor methods for specific target data
associations = await client.get_target_data_associations(target_id, min_score=0.1)
pathways = await client.get_target_data_pathways(target_id)
interactions = await client.get_target_data_interactions(target_id)
known_drugs = await client.get_target_data_drug_summaries(target_id)
expressions = await client.get_target_data_tissue_expression(target_id)
phenotypes = await client.get_target_data_mouse_phenotypes(target_id)
safety = await client.get_target_data_safety_liabilities(target_id)
constraints = await client.get_target_data_genetic_constraints(target_id)

# Drug-specific accessor
indications = await client.get_drug_indications(drug_name)
```

---

## ClinicalTrials.gov Data Structure

The `ClinicalTrialsClient` provides four public methods for clinical trial data:

| Method | Purpose | Returns |
|--------|---------|---------|
| `search_trials(drug, condition)` | Find trials for a drug-condition pair | `list[Trial]` |
| `detect_whitespace(drug, condition)` | Is this pair unexplored? | `WhitespaceResult` |
| `get_landscape(condition)` | Competitive landscape for a condition | `ConditionLandscape` |
| `get_terminated(query)` | Failed trials for a drug/condition | `list[TerminatedTrial]` |

### Trial

Core trial record parsed from ClinicalTrials.gov API `protocolSection`:

```
Trial
 |-- nct_id: str                    # NCT identifier (e.g. "NCT04375669")
 |-- title: str                     # Brief title of the study
 |-- brief_summary: str | None      # Study description
 |-- phase: str                     # "Phase 1", "Phase 2", "Phase 1/Phase 2", etc.
 |-- overall_status: str            # "Recruiting", "Completed", "Terminated", etc.
 |-- why_stopped: str | None        # Free text reason (only for Terminated/Withdrawn/Suspended)
 |-- conditions: list[str]          # Disease/conditions being studied
 |-- interventions: list[Intervention]
 |        |-- intervention_type: str    # "Drug", "Biological", "Device", "Procedure", etc.
 |        |-- intervention_name: str    # e.g. "Semaglutide"
 |        +-- description: str | None   # Dosing, administration details
 |-- sponsor: str                   # Lead sponsor organization (e.g. "Novo Nordisk")
 |-- collaborators: list[str]       # Partner organizations
 |-- enrollment: int | None         # Number of participants
 |-- start_date: str | None         # Study start date
 |-- completion_date: str | None    # Primary completion date
 |-- study_type: str                # "Interventional" or "Observational"
 |-- primary_outcomes: list[PrimaryOutcome]
 |        |-- measure: str              # What's being measured
 |        +-- time_frame: str | None    # e.g. "72 weeks"
 |-- results_posted: bool           # Whether results are available
 +-- references: list[str]          # PubMed IDs (PMIDs)
```

### WhitespaceResult

Result of whitespace detection — is this drug-condition pair unexplored?

```
WhitespaceResult
 |-- is_whitespace: bool            # True if no exact matches found
 |-- exact_match_count: int         # Trials with both drug AND condition
 |-- drug_only_trials: int          # Trials with drug (any condition)
 |-- condition_only_trials: int     # Trials with condition (any drug)
 +-- condition_drugs: list[ConditionDrug]  # Other drugs tested for this condition (up to 20)
          |-- nct_id: str
          |-- drug_name: str
          |-- condition: str
          |-- phase: str
          +-- status: str
```

### ConditionLandscape

Competitive landscape for a condition — all drug/biologic trials grouped by sponsor + drug:

```
ConditionLandscape
 |-- total_trial_count: int         # All trials fetched for condition
 |-- competitors: list[CompetitorEntry]   # Ranked by phase, then enrollment (top_n)
 |        |-- sponsor: str              # Lead sponsor organization
 |        |-- drug_name: str            # Primary drug intervention
 |        |-- drug_type: str | None     # "Drug" or "Biological"
 |        |-- max_phase: str            # Highest phase reached
 |        |-- trial_count: int          # Number of trials for this sponsor+drug
 |        |-- statuses: set[str]        # All statuses seen (Recruiting, Completed, etc.)
 |        |-- total_enrollment: int     # Sum of enrollment across trials
 |        +-- most_recent_start: str | None  # Latest start date
 |-- phase_distribution: dict[str, int]  # Count of trials per phase
 +-- recent_starts: list[dict]      # Trials started in last 2 years
```

### TerminatedTrial

A terminated, withdrawn, or suspended trial with stop classification:

```
TerminatedTrial
 |-- nct_id: str
 |-- title: str
 |-- drug_name: str | None          # Primary drug intervention
 |-- condition: str | None          # First condition listed
 |-- phase: str | None
 |-- why_stopped: str | None        # Free text reason from sponsor
 |-- stop_category: str             # Classified: safety, efficacy, business, enrollment, other, unknown
 |-- enrollment: int | None
 |-- sponsor: str | None
 |-- start_date: str | None
 |-- termination_date: str | None
 +-- references: list[str]          # PMIDs
```

### Stop Category Classification

The `stop_category` is derived from `why_stopped` using keyword matching:

| Keywords | Category |
|----------|----------|
| efficacy, futility, lack of efficacy, no benefit | `efficacy` |
| safety, adverse, toxicity, side effect | `safety` |
| enrollment, accrual, recruitment | `enrollment` |
| business, strategic, funding, commercial | `business` |
| (no match) | `other` |
| (no why_stopped text) | `unknown` |

### Filtering

`get_landscape` filters to drug/biologic interventions only:
1. Fetches all trials for the condition (no API-level filter)
2. Skips trials without Drug or Biological intervention type
3. Groups remaining by sponsor + drug
4. Ranks by phase (descending), then enrollment (descending)
5. Returns top N competitors (default 50)