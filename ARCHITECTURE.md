# Architecture

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query biomedical data sources and produce a repurposing report.

## Layers

```
CLI / API (FastAPI)
    |
Orchestrator Agent
    |
Specialist Agents: literature, clinical_trials, mechanism, safety
    |
Services: LLM, embeddings, disease normalization, PubMed query building, retrieval (RAG)
    |
Data Source Clients: OpenTargets (GraphQL), ClinicalTrials.gov (REST), PubMed (REST+XML), ChEMBL (REST), DrugBank (stub)
    |
Pydantic Models -- contract boundary; agents receive these, never raw dicts
    |
External APIs + PostgreSQL/pgvector (abstract cache + vector store)
```

## Source Layout (`src/indication_scout/`)

- **data_sources/** -- Async API clients extending `BaseClient` (retry/backoff, session lifecycle). Errors surface as `DataSourceError`.
- **models/** -- Pydantic `BaseModel` contracts per source: `model_open_targets.py`, `model_clinical_trials.py`, `model_pubmed_abstract.py`, `model_chembl.py`, `model_drug_profile.py`.
- **agents/** -- AI agents extending `BaseAgent` (single `async run()` method). All stubs.
- **services/** -- Business logic: `llm.py` (Anthropic SDK), `embeddings.py` (BioLORD-2023), `disease_normalizer.py`, `pubmed_query.py`, `retrieval.py` (RAG pipeline).
- **api/** -- FastAPI app with `/health` endpoint. Routes and schemas not yet defined.
- **db/** -- SQLAlchemy session factory.
- **sqlalchemy/** -- ORM models (`pubmed_abstracts` with pgvector embedding column, 768 dims).
- **prompts/** -- LLM prompt templates: `extract_organ_term.txt`, `expand_search_terms.txt`, `disease_synonyms.txt`.
- **utils/** -- Shared file-based cache utility (`cache_key`, `cache_get`, `cache_set`).
- **helpers/** -- Drug name normalization.
- **config.py** -- `pydantic_settings.BaseSettings` loaded from `.env`.
- **constants.py** -- URLs, timeouts, cache TTL, lookup maps.

## Data Flow

```
Drug name
    |
    v
build_drug_profile (Open Targets + ChEMBL) --> DrugProfile
    |
    v
expand_search_terms (LLM) --> PubMed keyword queries (5-10)
    |
    v
fetch_and_cache --> PubMed search --> fetch abstracts --> embed (BioLORD-2023) --> pgvector
    |
    v
semantic_search --> cosine similarity over pgvector --> top abstracts
    |
    v
synthesize (Claude) --> EvidenceSummary (not yet implemented)
```

## Key Patterns

- **BaseClient**: All data source clients use `async with Client() as c:` for session lifecycle. Provides `_rest_get()`, `_graphql()`, `_rest_get_xml()` with retry.
- **File-based cache**: SHA-256-keyed JSON files in `_cache/` dir, 5-day TTL. Used by Open Targets, ChEMBL, PubMed, disease normalizer, and retrieval service. Shared utility in `utils/cache.py`.
- **pgvector**: PostgreSQL + pgvector for PubMed abstract embeddings. BioLORD-2023 (768 dims) via `sentence-transformers`.
- **Pydantic defensive defaults**: All models ingesting external data use a `coerce_nones` model validator to handle API nulls.
- **DrugProfile**: Flat LLM-facing projection of `RichDrugData` -- name, synonyms, target gene symbols, mechanisms, ATC codes/descriptions, drug type.

## Infrastructure

- PostgreSQL 16 + pgvector via Docker (`docker-compose.yml`, port 5438)
- Alembic for database migrations
- LLM: Claude Sonnet 4.6 (main), Claude Haiku 4.5 (lightweight calls)
- Embedding: BioLORD-2023 (768-dim, loaded via `sentence-transformers`)

See `docs/` for detailed data contracts per client, and `docs/rag_details.md` for RAG pipeline specifications.
