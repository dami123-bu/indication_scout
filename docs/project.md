## IndicationScout — Codebase Overview

### What It Is

A **drug repurposing discovery tool** — an agentic system that takes a drug name (e.g. "baricitinib") and finds new therapeutic indications by stacking biological, clinical, and literature evidence from multiple biomedical data sources.

### Languages & Frameworks

| | |
|---|---|
| **Language** | Python 3.10+ (running 3.11) |
| **Async I/O** | `aiohttp`, `asyncio` |
| **Web** | `FastAPI` + `uvicorn` (minimal — health check only) |
| **Data contracts** | `pydantic` v2 |
| **Config** | `pydantic-settings` (loads `.env`) |
| **Data** | `numpy`, `pandas`, `scikit-learn` |
| **Testing** | `pytest`, `pytest-asyncio`, `pytest-cov` |
| **Code quality** | `black`, `ruff`, `mypy` (strict) |
| **Build** | `setuptools` |

---

### Project Structure

```
src/indication_scout/
├── data_sources/          # External API clients (the working layer)
│   ├── base_client.py     # Abstract async HTTP client — retry, rate limits, caching
│   ├── open_targets.py    # GraphQL client (741 lines) — drugs, targets, diseases
│   ├── clinical_trials.py # REST client (597 lines) — trials, whitespace, landscape
│   ├── pubmed.py          # E-utilities client (166 lines) — search, fetch articles
│   ├── chembl.py          # Stub (NotImplementedError)
│   └── drugbank.py        # Stub (NotImplementedError)
│
├── models/                # Pydantic data contracts (24+ classes)
│   ├── model_open_targets.py    # DrugData, TargetData, Association, etc. (251 lines)
│   ├── model_clinical_trials.py # Trial, WhitespaceResult, ConditionLandscape, etc. (134 lines)
│   └── model_pubmed.py          # PubMedArticle (16 lines)
│
├── agents/                # Agent layer (ALL STUBS — not yet built)
│   ├── base.py            # Abstract BaseAgent
│   ├── orchestrator.py    # ReAct coordinator
│   ├── literature.py      # RAG over PubMed
│   ├── clinical_trials.py # Trial analysis
│   ├── mechanism.py       # Mechanism of action
│   └── safety.py          # Safety analysis
│
├── services/              # Business logic layer (PLACEHOLDER — empty)
├── api/                   # FastAPI app (MINIMAL — /health only)
│   ├── main.py
│   ├── routes/
│   └── schemas/
├── config.py              # Settings singleton (DB URL, API keys, model names)
└── constants.py           # Timeouts, TTLs, URLs, keyword maps

tests/
├── conftest.py            # Root fixtures
├── unit/                  # No external calls (1,123 lines)
│   ├── test_base_client.py
│   ├── test_clinical_trials_helpers.py
│   ├── test_clinical_trials_models.py
│   ├── test_open_targets_models.py
│   └── test_pubmed.py
├── integration/           # Hit real APIs (1,546 lines)
│   ├── test_open_targets.py
│   ├── test_open_target_accessors.py
│   ├── test_clinical_trials.py
│   ├── test_pubmed.py
│   └── test_base_client.py
└── fixtures/

docs/                      # API contracts & architecture
├── ARCHITECTURE.md
├── api_clients.md
├── open_targets.md
├── clinical_trials.md
└── roadmap.md             # Placeholder

planning_docs/             # Sprint plan, implementation details, progress
notebooks/                 # 3 Jupyter notebooks (EDA, features, experiments)
scripts/                   # run_evaluation.py, seed_db.py (both stubs)
_cache/                    # Disk cache for API responses (JSON, 5-day TTL)
```

---

### Configuration Files

| File | Purpose |
|------|---------|
| pyproject.toml | Dependencies, build config, tool settings (black, ruff, mypy, pytest) |
| pytest.ini | `asyncio_mode = auto` |
| .env.example | Template: DB URL, API keys, model names |
| CLAUDE.md | Project rules for AI-assisted development |
| .gitignore | Excludes `_cache/`, `planning_docs/`, `.env` |

---

### Component Status

| Layer | Status | Lines |
|-------|--------|-------|
| **Data Sources** | **Done** | ~1,700 (3 working clients + 2 stubs) |
| **Data Models** | **Done** | ~400 (24+ Pydantic classes) |
| **Tests** | **Substantial** | ~2,700 (unit + integration, well-separated) |
| **Documentation** | **Good** | 5 docs covering architecture & API contracts |
| **Agents** | **Stub** | All 5 raise `NotImplementedError` |
| **Services** | **Empty** | Placeholder |
| **API** | **Minimal** | `/health` endpoint only |
| **CLI** | **Not built** | Entry point declared in pyproject.toml but no file |
| **RAG pipeline** | **Not built** | PubMed fetch exists, embedding + pgvector not yet |
| **Database** | **Not built** | Config references SQLite, no schema |

**Bottom line:** The data layer is solid and well-tested. Everything above it — agents, orchestration, RAG, services, UI — is the sprint work ahead per your sprint plan.
