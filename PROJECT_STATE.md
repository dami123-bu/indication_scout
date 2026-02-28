# IndicationScout — Project State

> Auto-updated. Do not edit manually. Run /memory to refresh.

## What This Is

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources and produce a repurposing report identifying candidate indications not yet approved or explored.

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| `data_sources/base_client.py` | Complete | Async HTTP client with retry/backoff, REST GET, GraphQL POST, XML GET |
| `data_sources/open_targets.py` | Complete | Full GraphQL client; drug, target, disease, disease synonyms; file-based cache |
| `data_sources/clinical_trials.py` | Complete | ClinicalTrials.gov v2 REST; 5 public methods; whitespace detection, landscape, terminated |
| `data_sources/pubmed.py` | Complete | PubMed ESearch/EFetch; search, count, fetch abstracts from XML |
| `data_sources/chembl.py` | Stub | `search_compound` and `get_activities` both raise `NotImplementedError` |
| `data_sources/drugbank.py` | Stub | `get_drug` and `get_interactions` both raise `NotImplementedError` |
| `models/model_open_targets.py` | Complete | Full Pydantic contract for all OT data types |
| `models/model_clinical_trials.py` | Complete | Full Pydantic contract for trial, whitespace, landscape, terminated |
| `models/model_pubmed_abstract.py` | Complete | `PubmedAbstract` model |
| `agents/orchestrator.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/literature.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/clinical_trials.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/mechanism.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/safety.py` | Stub | `run()` raises `NotImplementedError` |
| `services/llm.py` | Complete | `query_llm` and `query_small_llm` via Anthropic SDK |
| `services/disease_normalizer.py` | Complete | LLM normalization of disease terms for PubMed; blocklist guard; PubMed count verification |
| `services/pubmed_query.py` | Complete | Builds PubMed queries by normalizing disease name and combining with drug name |
| `services/retrieval.py` | Partial | `get_disease_synonyms` works; `fetch_and_cache`, `semantic_search`, `synthesize`, `expand_search_terms` all raise `NotImplementedError` |
| `sqlalchemy/pubmed_abstracts.py` | Complete | SQLAlchemy ORM model with pgvector embedding column (768 dims) |
| `db/session.py` | Complete | SQLAlchemy session factory; `get_db()` dependency |
| `api/main.py` | Partial | FastAPI app with `/health` endpoint only; routes/ and schemas/ subdirs are empty |
| `helpers/drug_helpers.py` | Complete | `normalize_drug_name` strips common salt suffixes |
| `runners/pubmed_runner.py` | Partial | Development script demonstrating OT client; uses `print()` (violates project rules) |
| `config.py` | Complete | Pydantic settings from `.env`; LLM model, DB URL, API keys |
| `constants.py` | Complete | Timeout, cache TTL, OT URL, interaction type map, stop-reason keywords |

## Architecture & Data Flow

The system is organized into five layers. Agents sit above data sources and communicate only through Pydantic models; they never receive raw API responses.

```
CLI / API
    |
Orchestrator (agents/orchestrator.py)  — stub
    |
Specialist agents: literature, clinical_trials, mechanism, safety  — all stubs
    |
Services layer: llm.py, disease_normalizer.py, pubmed_query.py, retrieval.py
    |
Data source clients: OpenTargetsClient, ClinicalTrialsClient, PubMedClient
    |
Pydantic models (models/)  — contract boundary; agents receive these, never raw dicts
    |
External APIs: Open Targets GraphQL, ClinicalTrials.gov v2 REST, PubMed EUtils
```

The database layer (PostgreSQL + pgvector) is used for caching PubMed abstracts with their embeddings. The embedding model is `FremyCompany/BioLORD-2023` (768 dims). The Open Targets client uses a separate file-based cache (`_cache/` dir, SHA-256-keyed JSON files, 5-day TTL) that does not require the database.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/indication_scout/data_sources/open_targets.py` | Most complex client; GraphQL queries are defined as module-level strings at the bottom of the file |
| `src/indication_scout/data_sources/clinical_trials.py` | `detect_whitespace()` runs 3 concurrent API calls; `get_landscape()` aggregates trials into a competitive map |
| `src/indication_scout/services/disease_normalizer.py` | LLM-driven disease term normalization with a blocklist guard against over-broad terms |
| `src/indication_scout/services/retrieval.py` | Intended RAG pipeline for PubMed (mostly stubs; only `get_disease_synonyms` works) |
| `src/indication_scout/sqlalchemy/pubmed_abstracts.py` | ORM model for the `pubmed_abstracts` pgvector table |
| `src/indication_scout/runners/pubmed_runner.py` | Development/exploration script, not part of production pipeline |
| `tests/integration/data_sources/test_open_targets.py` | Extensive integration suite with exact field assertions; doubles as API contract verification |
| `docs/open_targets.md` | Full data contract documentation for Open Targets client |
| `planning_docs/` | Sprint plans and technical implementation docs |

## Non-Obvious Patterns & Decisions

- The Open Targets client resolves drug names to ChEMBL IDs via a GraphQL search before fetching drug data. Names are case-insensitive and fuzzy; the first match wins.
- `get_rich_drug_data()` fetches all target data in parallel using `asyncio.gather`, benefiting from the file cache for repeated calls.
- `detect_whitespace()` runs three concurrent API calls (exact match, drug-only count, condition-only count) then only fetches full condition trials if whitespace exists. When whitespace exists, `condition_drugs` is populated with Phase 2+ competitors ranked by phase then active status.
- The ClinicalTrials v2 API uses `AREA[Phase]` syntax for phase filtering embedded in `query.term`, not a dedicated parameter.
- The `DiseaseSynonyms.all_synonyms` property deliberately excludes `broad` and `narrow` synonyms; it only returns `exact + related + parent_names`.
- The disease normalizer has a two-step LLM strategy: first normalize, then if PubMed hit count is below `MIN_RESULTS` (3), ask the LLM to generalize further. Both steps are blocked if the result is an over-generic term in `BROADENING_BLOCKLIST`.
- `pubmed_query.get_pubmed_query()` returns a list of queries (one per disease term when the normalized result contains `OR`), not a single string.
- The `stop_category` field on `TerminatedTrial` is a keyword-based pre-classification; the docstring notes that LLM refinement is intended to happen at the agent layer (not yet implemented).
- LLM model: `claude-sonnet-4-6` for main model, `claude-haiku-4-5-20251001` for lightweight calls (e.g. disease normalization).
- The `get_drug_competitors()` method on `OpenTargetsClient` has a `# TODO needs rework` comment; it is implemented but may produce inconsistent results.
- `api/routes/` and `api/schemas/` subdirectories exist but contain only `__init__.py` — no routes or schemas are defined yet.
- The CLI entry point `scout find -d "metformin"` is referenced in pyproject.toml scripts but the CLI module `indication_scout.cli.cli` does not exist yet.
- `runners/pubmed_runner.py` uses `print()` which violates the project rule of logging-only output.

## Known Issues / Caveats

- All five agents (`Orchestrator`, `LiteratureAgent`, `ClinicalTrialsAgent`, `MechanismAgent`, `SafetyAgent`) raise `NotImplementedError` in their `run()` methods — the agent layer is completely unimplemented.
- The RAG pipeline in `services/retrieval.py` is almost entirely stubbed: `fetch_and_cache`, `semantic_search`, `synthesize`, and `expand_search_terms` all raise `NotImplementedError`. Only `get_disease_synonyms` (which calls an LLM) works.
- `ChEMBLClient` and `DrugBankClient` are stubs.
- The CLI module referenced in `pyproject.toml` (`indication_scout.cli.cli`) does not exist.
- `tests/integration/data_sources/test_open_targets.py` contains two tests marked `# TODO rework` (`test_surfacing_pipeline`, `test_get_drug_target_competitors_semaglutide`) — they call the partially-implemented `get_drug_competitors()` method and may be fragile.
- `runners/pubmed_runner.py` uses `print()` instead of the `logging` module, which violates project rules.
- `MEMORY.md` is marked as deleted in git status (D MEMORY.md) — it has been removed and replaced by `PROJECT_STATE.md`.
- The `db/session.py` creates a new engine and session factory on every call to `get_db()` — there is no connection pooling singleton.
