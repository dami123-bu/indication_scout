# Findings

Important discoveries, decisions, and patterns encountered during development.
Each entry is dated and categorized.

---

## Project Rules

### Never remove TODO comments
- TODO comments must be left in place — do not remove them during code review, cleanup, or refactoring

---

## API & Data Sources

### PubMed model naming (2026-03-01)
- Model is `PubmedAbstract` (not `PubMedArticle`); method is `fetch_abstracts` (not `fetch_articles`)
- File: `models/model_pubmed_abstract.py`

### Disk cache is shared (2026-03-01)
- `utils/cache.py` is used by ALL clients and services, not just OpenTargets
- Any client or service can call `cache_get` / `cache_set` directly

### `.env.example` had wrong embedding model (2026-03-01)
- Was `text-embedding-3-small` (OpenAI); correct value is `FremyCompany/BioLORD-2023`
- Fixed in `.env.example`

---

## Models & Data Contracts

### `expand_search_terms` input type (2026-03-01)
- Takes `DrugProfile`, not `RichDrugData`
- `DrugProfile` is built from `RichDrugData` upstream via `DrugProfile.from_rich_drug_data()`

### `DrugData.atc_classifications` frequently omitted (2026-03-01)
- Field exists and is populated; easy to miss when documenting `DrugData`

### Pydantic `coerce_nones` validator (2026-03-01)
- Every model ingesting external data has a `model_validator(mode="before")` that coerces `None` → default
- Does **not** inherit to nested models — must be added to every model in the hierarchy

---

## Architecture & Patterns

### `get_drug_competitors` layering fix (2026-03-11)
- `OpenTargetsClient.get_drug_competitors` was calling `merge_duplicate_diseases` (which calls the LLM via `query_small_llm`), making a data source client depend on a service — an architectural inversion.
- Fix: client now returns `CompetitorRawData` (raw `diseases` dict + `drug_indications` list). `RetrievalService.get_drug_competitors` owns the merge/remove/sort/cache logic and calls `merge_duplicate_diseases` directly.
- `drug_indications` must be included in `CompetitorRawData` because `merge_duplicate_diseases` passes it to the LLM prompt so the model knows which diseases are confirmed approved indications — context required for correct merging decisions.
- Cache (keyed by `drug_name`) now lives in the service, not the client.

### PubMed date filter bug (2026-03-11)
- `PubMedClient.search()` and `get_count()` had `params["datetype_maxdate"]` — a single merged key that NCBI does not recognise, so date filtering was silently ignored.
- Fix: two separate params — `datetype=pdat` and `maxdate=YYYY/MM/DD` (slash-separated, as required by NCBI eutils).

### `markers.py` location (2026-03-01)
- Lives at `src/indication_scout/markers.py` (package root), not in a subdirectory
- Easy to omit from project structure listings

### CLAUDE.md discrepancy (2026-03-01)
- CLAUDE.md references `model_pubmed.py` / `PubMedArticle` — both are wrong
- Actual file/class: `model_pubmed_abstract.py` / `PubmedAbstract`
- CLAUDE.md is read-only; this cannot be fixed there

---

## Observability & Tooling

### W&B integration (2026-03-14)
- `wandb` added as a runtime dependency in `pyproject.toml`.
- `wandb.init` / `wandb.finish` encapsulated in a reusable `@wandb_run(project, tracked_param)` decorator in `utils/wandb_utils.py`. Uses `inspect.signature` + `bind` to extract the tracked param by name, so it works regardless of positional vs keyword call.
- `RetrievalService.semantic_search` logs a `wandb.Table` (pmid, title, similarity) plus scalar metrics per disease when `wandb.run` is active. Guard is `if wandb.run:` — no-op when W&B is not initialised.
- Metric keys are namespaced as `semantic_search/{disease_key}/...` where spaces are replaced with underscores — spaces in keys broke W&B table rendering.
- `WANDB_API_KEY` read from env; declared as `wandb_api_key` field in `config.py` (was previously a typo: `wand_api_key`).

### `get_drug_competitors` cache shape mismatch bug (2026-03-14)
- After the layering fix in commit `3719cdb`, `OpenTargetsClient.get_drug_competitors` cache-hit path returned `{disease: set(drugs)}` (old flat shape) instead of `CompetitorRawData`. `RetrievalService` then failed with `KeyError` on `raw["diseases"]`.
- Decision: do not silently fix the shape mismatch — user deleted stale cache entries to resolve. The broken cache-hit return path (`line 133`) remains and will raise `KeyError` if a stale cache entry is hit again. Raising is the desired behaviour.

---

## Prompt Engineering

### `expand_search_terms` axis balance (2026-03-01)
- With 3 target gene queries (PRKAA1, PRKAA2, STK11) on a 9-query cap, one axis dominates
- Mitigation: prompt could cap target gene queries at 2 — or let dedup in `fetch_and_cache` handle overlap
- Decision pending; see `docs/notes.md`