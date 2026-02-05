# IndicationScout Architecture

**Version:** 0.1.0 (pre-alpha)
**Python:** >=3.10
**Build:** setuptools

IndicationScout is a toolkit for drug repurposing and indication discovery. Given a drug, it gathers evidence from multiple biomedical data sources, scores the strength of that evidence for potential new indications, and produces a structured report.

---

## Directory Structure

```
src/indication_scout/
├── __init__.py                  # Package root, defines __version__
├── config.py                    # Application settings (env-based)
│
├── models/                      # Pydantic domain models
│   ├── target.py                # Target (gene/protein a drug acts on)
│   ├── indication.py            # Indication (base), DiseaseIndication (with ontology IDs)
│   └── drug.py                  # Drug, DrugActivity (mechanism-target-indication link)
│
├── data/                        # Implemented external API clients
│   └── opentargets.py           # OpenTargetsClient (GraphQL, async httpx)
│
├── data_sources/                # Planned external API clients (all stubbed)
│   ├── pubmed.py                # PubMedClient
│   ├── clinicaltrials_gov.py    # ClinicalTrialsClient
│   ├── chembl.py                # ChEMBLClient
│   └── drugbank.py              # DrugBankClient
│
├── agents/                      # Multi-agent analysis layer (all stubbed)
│   ├── base.py                  # BaseAgent ABC
│   ├── mechanism.py             # MechanismAgent
│   ├── literature.py            # LiteratureAgent
│   ├── clinical_trials.py       # ClinicalTrialsAgent
│   ├── safety.py                # SafetyAgent
│   └── orchestrator.py          # Orchestrator (coordinates agents)
│
├── services/                    # Business logic (empty)
│
├── db/                          # Persistence layer (all stubbed)
│   ├── connection.py            # Database async connection manager
│   ├── repositories.py          # DrugRepository, IndicationRepository, ReportRepository
│   └── migrations/              # Empty
│
└── api/                         # FastAPI application
    ├── main.py                  # App factory, health check endpoint
    ├── routes/                  # Empty
    └── schemas/                 # Empty

tests/
├── conftest.py                  # Shared fixtures (sample_drug, sample_indication)
├── unit/                        # Empty
├── integration/
│   └── test_opentargets.py      # Live API tests for OpenTargetsClient
└── fixtures/                    # Empty

scripts/
├── run_evaluation.py            # Placeholder
└── seed_db.py                   # Placeholder
```

---

## Domain Models

All domain models are Pydantic `BaseModel` subclasses.

### Target (`models/target.py`)

A gene or protein that a drug may act on.

```
Target
├── id: str                      # Source-independent identifier
├── ncbi_id: str | None          # NCBI gene ID
├── ensembl_id: str | None       # Ensembl gene ID, e.g. "ENSG00000146648"
└── symbol: str                  # Gene symbol, e.g. "EGFR"
```

### Indication hierarchy (`models/indication.py`)

Represents anything a drug can be used to treat — diseases, symptoms, syndromes, or other clinical presentations (e.g. "fever", "puffy eyes").

```
Indication (base)
├── id: str
└── name: str

DiseaseIndication (extends Indication)
├── efo_id: str | None           # EFO ID (Open Targets)
└── mondo_id: str | None         # MONDO ID (Monarch Initiative)
```

### Drug and DrugActivity (`models/drug.py`)

A pharmaceutical compound and its mechanism-target-indication relationships.

```
Drug
├── id: str                      # Source-independent identifier
├── chembl_id: str | None
├── drugbank_id: str | None
├── generic_name: str
├── brand_name: str | None
├── description: str | None
├── drug_type: str | None
├── max_clinical_phase: int | None
└── activities: list[DrugActivity]
      ├── description: str | None      # e.g. "Cyclooxygenase inhibitor"
      ├── target: Target | None        # e.g. Target(symbol="PTGS2")
      └── indication: Indication | None
```

All DrugActivity fields are optional to allow partially populated entries from different data sources. A single Drug can have multiple DrugActivity entries — one per mechanism-target-indication combination.

---

## OpenTargetsClient (the only implemented data client)

**Location:** `src/indication_scout/data/opentargets.py`
**Protocol:** GraphQL over HTTPS (`https://api.platform.opentargets.org/api/v4/graphql`)
**HTTP client:** `httpx.AsyncClient` (async, 30s timeout)

**Note:** This client currently imports deleted models (`DrugMechanism`, `ApprovedIndication`) and uses old `Drug` field names. It needs to be updated to use the current domain models.

### Implemented methods

| Method | What it does | Returns |
|---|---|---|
| `get_drug(chembl_id)` | Fetches drug info, mechanisms of action, and approved indications | `Drug \| None` |
| `search(term, entity_type)` | Free-text search across drugs, diseases, or targets | `list[dict]` |
| `resolve_id(term, entity_type)` | Convenience wrapper: returns the top search hit's ID | `str \| None` |

### Stubbed methods (signature only, no implementation)

| Method | Intended purpose |
|---|---|
| `get_drug_mechanisms(chembl_id)` | Detailed mechanism-of-action data |
| `get_drug_indications(chembl_id)` | Approved indications with clinical phase |
| `get_disease_targets(efo_id, min_score)` | Targets associated with a disease above a score threshold |

---

## Agent Layer

All agents inherit from `BaseAgent` (an ABC with a single `async run(input_data) -> dict` method). None are implemented.

| Agent | Intended role |
|---|---|
| `MechanismAgent` | Determine if a drug's target is relevant to a candidate disease |
| `LiteratureAgent` | Search PubMed for published evidence of the drug-disease link |
| `ClinicalTrialsAgent` | Check ClinicalTrials.gov for active or completed trials |
| `SafetyAgent` | Assess safety signals and contraindications |
| `Orchestrator` | Coordinate the above agents, collect results |

---

## Configuration

`config.py` uses `pydantic-settings` to load from environment variables (with `.env` file support):

| Setting | Purpose |
|---|---|
| `database_url` | SQLite connection string |
| `openai_api_key` | For LLM-based scoring / rationale generation |
| `pubmed_api_key` | PubMed E-utilities API key |
| `llm_model` | Model name (default: gpt-4) |
| `embedding_model` | Embedding model name |
| `debug`, `log_level` | Runtime behavior |

---

## Database Layer

Stubbed with async interface patterns:

- **`Database`** (`db/connection.py`): Connection manager with `connect()`, `disconnect()`, and `session()` context manager. SQLite-backed.
- **`DrugRepository`**: CRUD for Drug objects
- **`IndicationRepository`**: CRUD for Indication objects
- **`ReportRepository`**: Save and retrieve analysis reports (references a `Report` model that does not yet exist)

---

## API Layer

`FastAPI` application in `api/main.py`. Currently only exposes:

```
GET /health  →  {"status": "healthy", "version": "0.1.0"}
```

`api/routes/` and `api/schemas/` directories exist but are empty.

---

## Entry Points

| Entry point | Location | Status |
|---|---|---|
| CLI (`scout` command) | `indication_scout.cli.cli:main` (registered in pyproject.toml) | Module does not exist |
| FastAPI | `src/indication_scout/api/main.py` | Health check only |
| Streamlit | Referenced in CLAUDE.md as `app.py` | Does not exist |

---

## Testing

**Framework:** pytest with `pytest-asyncio`
**Config:** `pyproject.toml [tool.pytest.ini_options]`, verbose output, short tracebacks

### Integration tests (`tests/integration/test_opentargets.py`)

Hit the live Open Targets API. Two test classes:

- **`TestResolveId`** (5 tests): Verifies `resolve_id` returns correct ID prefixes for drugs (CHEMBL), diseases (EFO/MONDO/Orphanet), and targets (ENSG). Tests a nonsense query returns `None`.
- **`TestGetDrug`** (5 tests): Verifies `get_drug` returns a populated `Drug` model for Aspirin (CHEMBL25) and Metformin (CHEMBL1431), including mechanisms and indications. Tests a nonexistent ID returns `None`.

### Unit tests

None yet.

### Fixtures (`tests/conftest.py`)

- `sample_drug`: Acetaminophen dict
- `sample_indication`: Pain dict
- `sample_disease`: Multiple Sclerosis dict (missing `@pytest.fixture` decorator)

---

## Dependencies

### Runtime
| Package | Purpose |
|---|---|
| `httpx` | Async HTTP client (used by OpenTargetsClient, not listed in pyproject.toml but imported) |
| `pydantic` / `pydantic-settings` | Model validation, settings |
| `fastapi` / `uvicorn` | API server |
| `click` | CLI framework |
| `numpy`, `pandas`, `scikit-learn` | Declared but not yet used |

### Dev
`pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `pre-commit`

---

## Known Issues

1. **`opentargets.py` uses stale imports**: Imports `DrugMechanism` (from deleted `mechanism.py`) and `ApprovedIndication` (removed from `indication.py`). Also references old `Drug` field names (`name`, `mechanisms`, `indications`). Will raise `ImportError` if imported.

2. **`httpx` not in pyproject.toml dependencies**: `opentargets.py` imports `httpx` but it is not listed in `[project.dependencies]`.

3. **`pytest-asyncio` not in dev dependencies**: Integration tests use `pytest.mark.asyncio` but `pytest-asyncio` is not listed in `[project.optional-dependencies.dev]`.

4. **Missing `@pytest.fixture` decorator**: `sample_disease` in `tests/conftest.py` is a plain function, not a fixture.

5. **CLI module does not exist**: `pyproject.toml` registers `scout = "indication_scout.cli.cli:main"` but no `cli/` package exists.

6. **Missing `Report` model**: `db/repositories.py` references a `Report` model that does not exist.
