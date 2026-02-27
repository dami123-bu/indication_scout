# IndicationScout — Session Memory

> Kept at project root. Update via `/memory` or by asking Claude to refresh it.
> Last updated: 2026-02-27

---

## Project Overview

Drug repurposing discovery tool. A drug name goes in; coordinated AI agents query multiple biomedical data sources and produce a repurposing report.

- Python 3.11, async (aiohttp), FastAPI, Pydantic v2, pytest
- Data layer is solid and well-tested; agents/services/API are stubs

---

## Key Paths

| Path | Purpose |
|------|---------|
| `src/indication_scout/` | All source code |
| `src/indication_scout/data_sources/` | Async API clients |
| `src/indication_scout/models/` | Pydantic models |
| `src/indication_scout/agents/` | AI agents (all stubs) |
| `src/indication_scout/services/` | Business logic (scaffolded) |
| `src/indication_scout/api/` | FastAPI app |
| `src/indication_scout/config.py` | pydantic-settings, loads `.env` |
| `src/indication_scout/constants.py` | URLs, limits, lookup maps |
| `tests/unit/` | No-network unit tests |
| `tests/integration/` | Real API integration tests |
| `docs/` | Architecture, agent spec, roadmap |
| `planning_docs/` | Sprint plan, progress tracking |

---

## Data Source Clients

| Client | Status | Notes |
|--------|--------|-------|
| `OpenTargetsClient` | Complete | GraphQL; drug/target/disease; file-based cache (5-day TTL in `_cache/`) |
| `ClinicalTrialsClient` | Complete | REST v2; search, whitespace detection, landscape, terminated trials |
| `PubMedClient` | Complete | REST+XML; search + fetch abstracts; batches 100/call |
| `FDAClient` | In progress | openFDA FAERS adverse events; models/constants/config done; client + tests remain (on `fda` branch) |
| `ChEMBLClient` | Stub | `NotImplementedError` on all methods |
| `DrugBankClient` | Stub | `NotImplementedError` on all methods |

All clients extend `BaseClient` (retry/backoff, 3 retries on 429/5xx, async context manager).

---

## Pydantic Models

| File | Key Models |
|------|-----------|
| `model_open_targets.py` | `TargetData`, `DrugData`, `RichDrugData`, `DrugSummary`, `Indication`, `DrugWarning`, `SafetyLiability`, `Association`, `Pathway`, `Interaction`, `TissueExpression`, `MousePhenotype`, `DiseaseSynonyms` |
| `model_clinical_trials.py` | `Trial`, `WhitespaceResult`, `ConditionLandscape`, `TerminatedTrial`, `CompetitorEntry` |
| `model_pubmed_abstract.py` | `PubmedAbstract` (pmid, title, abstract, authors, journal, pub_date, mesh_terms, keywords) |

---

## Agents (all stubs)

| Agent | Intended Role |
|-------|--------------|
| `Orchestrator` | Coordinates all specialist agents; main entry point |
| `LiteratureAgent` | PubMed search + RAG reranking |
| `ClinicalTrialsAgent` | Trial search, whitespace detection, terminated analysis |
| `MechanismAgent` | Drug MoA vs. target-disease pathways |
| `SafetyAgent` | FDA FAERS adverse events + safety scoring |

All extend `BaseAgent` with `async def run(input_data: dict) -> dict`. All raise `NotImplementedError`.

---

## API

- Only `/health` endpoint is implemented
- `api/routes/` and `api/schemas/` exist but are empty

---

## Config (`config.py`)

LLM defaults: `llm_model = "claude-sonnet-4-6"`, `small_llm_model = "claude-haiku-4-5-20251001"`, `embedding_model = "text-embedding-3-small"`

API keys in `.env`: `openai_api_key`, `anthropic_api_key`, `pubmed_api_key`, `ncbi_api_key`, `openfda_api_key`

---

## Test Conventions

- Plain functions (`def test_...`), no classes
- `pytest-asyncio` with `asyncio_mode = auto`
- Unit tests: no network, inline fixtures
- Integration tests: real APIs, never run automatically
- Assert all fields with actual values, not just types/existence

**Known pre-existing failure:** `test_search_trials_drug_only` — expects NCT04971785 in top 50 semaglutide results; API no longer returns it there. Not a code bug.

---

## User Preferences

- Step-by-step implementation; review plan before each step
- Do not start implementation without explicit instruction
- Research all fields/values before including them in plans
- Concise responses; no over-engineering
- Never fabricate scientific/clinical values — return `None` if unavailable

---

## What's Next (per sprint plan)

1. RAG pipeline + Path 2 discovery (drug class analogy)
2. Agents implemented as LangGraph tools
3. Orchestrator + report generation
4. Evaluation framework + Streamlit UI
5. FDA FAERS client (merge `fda` branch to main first)
