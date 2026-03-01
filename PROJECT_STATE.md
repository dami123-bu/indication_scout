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
| `data_sources/pubmed.py` | Complete | PubMed ESearch/EFetch; search (cached), count, fetch abstracts from XML; file-based cache on `search()` |
| `data_sources/chembl.py` | Partial | `get_molecule(chembl_id)` complete; `get_atc_description(atc_code)` planned (Step 2 of current PLAN.md) |
| `data_sources/drugbank.py` | Stub | `get_drug` and `get_interactions` both raise `NotImplementedError` |
| `models/model_chembl.py` | Partial | `MoleculeData` complete; `ATCDescription` model planned (Step 1 of current PLAN.md) |
| `models/model_open_targets.py` | Complete | Full Pydantic contract for all OT data types |
| `models/model_clinical_trials.py` | Complete | Full Pydantic contract for trial, whitespace, landscape, terminated |
| `models/model_pubmed_abstract.py` | Complete | `PubmedAbstract` model |
| `models/model_drug_profile.py` | Not started | `DrugProfile` + `from_rich_drug_data` factory planned (Step 3 of current PLAN.md) |
| `agents/orchestrator.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/literature.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/clinical_trials.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/mechanism.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/safety.py` | Stub | `run()` raises `NotImplementedError` |
| `services/llm.py` | Complete | `query_llm` and `query_small_llm` via Anthropic SDK |
| `services/disease_normalizer.py` | Complete | LLM normalization; blocklist guard; PubMed count verification; file-based cache for both LLM results and PubMed counts |
| `services/pubmed_query.py` | Complete | Builds PubMed queries by normalizing disease name and combining with drug name |
| `services/retrieval.py` | Partial | `get_disease_synonyms` works; `fetch_and_cache`, `semantic_search`, `synthesize` stub; `expand_search_terms` + `extract_organ_term` planned (Steps 5–7 of current PLAN.md) |
| `sqlalchemy/pubmed_abstracts.py` | Complete | SQLAlchemy ORM model with pgvector embedding column (768 dims) |
| `db/session.py` | Complete | SQLAlchemy session factory; `get_db()` dependency |
| `api/main.py` | Partial | FastAPI app with `/health` endpoint only; `api/routes/` and `api/schemas/` subdirs contain only `__init__.py` |
| `utils/cache.py` | Complete | Shared file-based cache utility: `cache_key`, `cache_get`, `cache_set`; used by all data source clients and services |
| `helpers/drug_helpers.py` | Complete | `normalize_drug_name` strips common salt suffixes |
| `scripts/session.py` | Complete | Session file manager: `startup` (create/rotate/print) and `append` subcommands; rotation to `session_bak/` with 5-file cap |
| `scripts/open_target_pipeline.py` | Complete | Exploratory pipeline: fetches bupropion competitors, builds PubMed queries, fetches abstracts; proper async with client context managers, `asyncio.run` under `__main__`, uses logging |
| `runners/pubmed_runner.py` | Partial | Development/exploration script; uses `print()` which violates project rules |
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
Data source clients: OpenTargetsClient, ClinicalTrialsClient, PubMedClient, ChEMBLClient
    |
Pydantic models (models/)  — contract boundary; agents receive these, never raw dicts
    |
External APIs: Open Targets GraphQL, ClinicalTrials.gov v2 REST, PubMed EUtils, ChEMBL REST
```

The database layer (PostgreSQL + pgvector) is used for caching PubMed abstracts with their embeddings. The embedding model is `FremyCompany/BioLORD-2023` (768 dims), loaded via `sentence-transformers` (added to runtime deps in `pyproject.toml`). The Open Targets client and the disease normalizer service both use a separate file-based cache (`_cache/` dir, SHA-256-keyed JSON files, 5-day TTL) that does not require the database.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/indication_scout/data_sources/open_targets.py` | Most complex client; GraphQL queries are defined as module-level strings at the bottom of the file |
| `src/indication_scout/data_sources/clinical_trials.py` | `detect_whitespace()` runs 3 concurrent API calls; `get_landscape()` aggregates trials into a competitive map |
| `src/indication_scout/services/disease_normalizer.py` | LLM-driven disease term normalization; two-step strategy (normalize then verify/broaden); uses `cache_get`/`cache_set` from `utils/cache.py` |
| `src/indication_scout/services/retrieval.py` | RAG pipeline; only `get_disease_synonyms` works; `expand_search_terms` + `extract_organ_term` next to implement |
| `src/indication_scout/sqlalchemy/pubmed_abstracts.py` | ORM model for the `pubmed_abstracts` pgvector table |
| `scripts/open_target_pipeline.py` | Exploratory async pipeline script; not part of production path |
| `runners/pubmed_runner.py` | Development/exploration script; uses `print()` in violation of project rules |
| `tests/integration/data_sources/test_open_targets.py` | Extensive integration suite with exact field assertions; doubles as API contract verification |
| `tests/integration/data_sources/test_pubmed_query.py` | Parametrized integration tests for `get_pubmed_query`; 5 drug-disease pairs; asserts query structure and disease keyword presence |
| `tests/integration/llm/test_retrieval.py` | Integration tests for retrieval service; `test_get_disease_synonyms` active; `expand_search_terms` stub to be uncommented when implemented |
| `docs/open_targets.md` | Full data contract documentation for Open Targets client |
| `docs/chembl.md` | Full data contract documentation for ChEMBL client (method, field mapping, agent usage) |
| `docs/api_clients.md` | Quick-reference guide for all data source clients; includes ChEMBLClient section |
| `docs/rag_details.md` | RAG pipeline implementation details; disease normalizer strategy documented in section 4 |
| `src/indication_scout/utils/cache.py` | Shared file-based cache utility; `cache_key`, `cache_get`, `cache_set` |
| `planning_docs/` | Sprint plans and technical implementation docs |

## Non-Obvious Patterns & Decisions

- The Open Targets client resolves drug names to ChEMBL IDs via a GraphQL search before fetching drug data. Names are case-insensitive and fuzzy; the first match wins.
- `get_rich_drug_data()` fetches all target data in parallel using `asyncio.gather`, benefiting from the file cache for repeated calls.
- `RichDrugData` (not `DrugData`) is the canonical input type for any agent or service that needs a complete drug profile. `DrugData` contains only drug-level metadata (name, synonyms, ATC codes, mechanisms). `RichDrugData` bundles `DrugData` with full `TargetData` for every target (pathways, disease associations, interactions, expression, safety liabilities). Always use `RichDrugData` when downstream code needs target-level context.
- `DrugProfile` (planned, `models/model_drug_profile.py`) is a flat LLM-facing projection of `RichDrugData`: name, synonyms, target gene symbols, mechanisms of action, ATC codes, ATC descriptions (human-readable from `/atc_class/{code}.json`), drug type. Built via `DrugProfile.from_rich_drug_data(rich, atc_descriptions)`. Used as the typed input to `expand_search_terms`.
- `expand_search_terms` generates diverse PubMed queries across 5 axes (drug name, drug class, mechanism+organ, target gene, synonym) using per-axis caps (5–10 total). Organ term is pre-extracted via a separate Haiku call (`extract_organ_term`). Both functions cache results.
- `detect_whitespace()` runs three concurrent API calls (exact match, drug-only count, condition-only count) then only fetches full condition trials if whitespace exists. When whitespace exists, `condition_drugs` is populated with Phase 2+ competitors ranked by phase then active status.
- The ClinicalTrials v2 API uses `AREA[Phase]` syntax for phase filtering embedded in `query.term`, not a dedicated parameter.
- The `DiseaseSynonyms.all_synonyms` property deliberately excludes `broad` and `narrow` synonyms; it only returns `exact + related + parent_names`.
- The disease normalizer has a two-step LLM strategy: first normalize, then if PubMed hit count is below `MIN_RESULTS` (3), ask the LLM to generalize further. Both steps are blocked if the result is an over-generic term in `BROADENING_BLOCKLIST`.
- The disease normalizer caches two types of results in `_cache/`: LLM-normalized terms under namespace `"disease_norm"` (key: `raw_term`) and PubMed result counts under namespace `"pubmed_count"` (key: full query string). Both use `cache_get`/`cache_set` from `utils/cache.py` with `DEFAULT_CACHE_DIR` passed explicitly.
- `PubMedClient.search()` now caches PMID lists under namespace `"pubmed_search"` (key: `query + max_results + date_before`) using the same file-based SHA-256 pattern. `get_count()` and `fetch_abstracts()` are not cached.
- `pubmed_query.get_pubmed_query()` returns a list of queries (one per disease term when the normalized result contains `OR`), not a single string.
- The `stop_category` field on `TerminatedTrial` is a keyword-based pre-classification; the docstring notes that LLM refinement is intended to happen at the agent layer (not yet implemented).
- LLM model: `claude-sonnet-4-6` for main calls, `claude-haiku-4-5-20251001` for lightweight calls (e.g. disease normalization).
- The `get_drug_competitors()` method on `OpenTargetsClient` has a `# TODO needs rework` comment; it is implemented but may produce inconsistent results.
- `api/routes/` and `api/schemas/` subdirectories exist but contain only `__init__.py` — no routes or schemas are defined yet.
- The CLI entry point `scout find -d "metformin"` is referenced in pyproject.toml scripts but the CLI module `indication_scout.cli.cli` does not exist yet.
- File-based cache utility (`cache_get`, `cache_set`, `cache_key`) lives in `utils/cache.py` and is shared by all callers. All data source clients and services import from there — no per-module duplication. Callers pass `cache_dir` explicitly; the utility never silently no-ops. New cache namespaces planned: `"atc_description"`, `"organ_term"`, `"expand_search_terms"`.

## Known Issues / Caveats

- All five agents (`Orchestrator`, `LiteratureAgent`, `ClinicalTrialsAgent`, `MechanismAgent`, `SafetyAgent`) raise `NotImplementedError` in their `run()` methods — the agent layer is completely unimplemented.
- The RAG pipeline in `services/retrieval.py` is almost entirely stubbed: `fetch_and_cache`, `semantic_search`, `synthesize` all raise `NotImplementedError`. `expand_search_terms` is actively being planned (current PLAN.md).
- `DrugBankClient` is a stub (`get_drug` and `get_interactions` raise `NotImplementedError`).
- `ChEMBLClient` has unit tests for `get_molecule` but no integration tests yet; `get_atc_description` not yet implemented.
- The CLI module referenced in `pyproject.toml` (`indication_scout.cli.cli`) does not exist.
- `tests/integration/data_sources/test_open_targets.py` contains two tests marked `# TODO rework` (`test_surfacing_pipeline`, `test_get_drug_target_competitors_semaglutide`) — they call the partially-implemented `get_drug_competitors()` method and may be fragile.
- `runners/pubmed_runner.py` uses `print()` instead of the `logging` module, which violates project rules.
- `tests/integration/data_sources/test_pubmed_query.py` contains two tests marked `# TODO delete` (`test_get_single_pubmed_query_returns_drug_and_term`, `test_get_single_disease_synonym`) — these are superseded by the parametrized suite but have not been removed yet.
- `db/session.py` creates a new engine and session factory on every call to `get_db()` — there is no connection pooling singleton.
