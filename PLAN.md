# IndicationScout — Development Plan

## System Overview

IndicationScout is an agentic drug repurposing system. A drug name goes in; AI agents query multiple biomedical sources and produce a repurposing report with grounded evidence.

---

## Current State

| Layer | Status | Notes |
|-------|--------|-------|
| `data_sources/` | Complete | PubMed, Open Targets, ClinicalTrials.gov live; ChEMBL and DrugBank are stubs (`NotImplementedError`) |
| `models/` | Complete | Pydantic contracts for Open Targets, ClinicalTrials, PubMed abstracts |
| `agents/` | Scaffolded | All 5 agents defined; all raise `NotImplementedError` (orchestrator, literature, clinical_trials, mechanism, safety) |
| `services/retrieval.py` | Scaffolded | `fetch_and_cache`, `semantic_search`, `synthesize`, `expand_search_terms` all raise `NotImplementedError` |
| `services/llm.py` | Unknown | Needs review |
| `services/pubmed_query.py` | Partial | PubMed query builder for RAG term expansion |
| `sqlalchemy/pubmed_abstracts.py` | Partial | ORM model for abstract storage; pgvector integration pending |
| `db/` | Scaffolded | `Base` (DeclarativeBase) and `session.py` (engine + session factory) exist; no migrations |
| `api/` | Scaffolded | Health check endpoint only; `routes/` and `schemas/` are empty |
| `helpers/drug_helpers.py` | Partial | Drug name normalization utilities |

---

## Phases

### Phase 1 — Core Data Pipeline
**Status: Complete (PubMed, Open Targets, ClinicalTrials.gov)**

- [x] `OpenTargetsClient` — GraphQL queries, disk-based caching, full drug/target/disease models
- [x] `ClinicalTrialsClient` — trial search, whitespace detection, condition landscape, terminated trials
- [x] `PubMedClient` — keyword search, abstract fetch, XML parsing, `PubmedAbstract` model
- [ ] `ChEMBLClient` — stub only
- [ ] `DrugBankClient` — stub only

### Phase 2 — RAG / Literature Retrieval Pipeline
**Status: Design complete (see `docs/rag.md`), implementation scaffolded**

The RAG pipeline solves the core problem: PubMed keyword search returns noisy results. Embeddings + reranking surfaces the papers that are actually relevant to a drug-disease pair.

Steps:
1. **Expand search terms** (`services/retrieval.py::expand_search_terms`) — LLM generates diverse PubMed queries from drug+disease (using full Drug object: mechanism, drug class, ATC codes, synonyms)
2. **Fetch and cache** (`services/retrieval.py::fetch_and_cache`) — hit PubMed for each query, embed new abstracts with BioLORD-2023 (768-dim), store in pgvector (deduplicate by PMID)
3. **Semantic search** (`services/retrieval.py::semantic_search`) — embed drug+disease query, cosine similarity over pgvector → top 20
4. **Re-rank** — cross-encoder or LLM reranker → top 5
5. **Synthesize** (`services/retrieval.py::synthesize`) — Claude generates structured `EvidenceSummary` with every claim traced to a PMID

Infrastructure required:
- PostgreSQL + pgvector extension
- BioLORD-2023 embedding model
- `sqlalchemy/pubmed_abstracts.py` ORM model wired to pgvector column
- Alembic migrations for `pubmed_abstracts` table

### Phase 3 — Agent Layer
**Status: Scaffolded**

All agents raise `NotImplementedError`. Implementation order:
1. `LiteratureAgent` — consumes RAG output (`EvidenceSummary`), produces structured literature analysis
2. `ClinicalTrialsAgent` — queries trial landscape, flags whitespace and terminated trials
3. `MechanismAgent` — mechanism-of-action reasoning using Open Targets target data
4. `SafetyAgent` — safety signal analysis
5. `Orchestrator` — coordinates all specialists, merges evidence, produces final report

### Phase 4 — API Layer
**Status: Scaffolded**

- Health check endpoint exists (`GET /health`)
- `routes/` and `schemas/` are empty
- Planned: `POST /scout` → accepts drug name, returns repurposing report

### Phase 5 — DB Persistence
**Status: Scaffolded**

- SQLAlchemy `Base` and `session.py` exist
- No migrations yet
- Alembic setup needed alongside pgvector table for Phase 2

---

## Open Questions

- `EvidenceSummary` model: needs to be defined in `models/` before Phase 2 synthesis step can be implemented
- Re-ranking strategy: cross-encoder (e.g. `ms-marco-MiniLM`) vs. LLM reranker — not yet decided
- ChEMBL / DrugBank: determine if these are needed before Phase 3 agents are implemented
- Report format: final output schema for the repurposing report is not yet defined
