# Findings

Important discoveries, decisions, and patterns encountered during development.
Each entry is dated and categorized.

---

## Project Rules

### Never remove TODO comments
- TODO comments must be left in place — do not remove them during code review, cleanup, or refactoring


### Pydantic `coerce_nones` validator (2026-03-01)
- Every model ingesting external data has a `model_validator(mode="before")` that coerces `None` → default
- Does **not** inherit to nested models — must be added to every model in the hierarchy

### ChEMBL ID is the sole drug identifier (2026-04-16)
- `DrugData` and `DrugProfile` carry only `chembl_id`; `name`, `synonyms`, `trade_names` removed
- All drug names (pref_name, synonyms, trade names) are fetched via `get_all_drug_names(chembl_id)` — single source of truth, cached per ChEMBL ID
- `get_all_drug_names` returns a list; `[0]` is the pref_name, `[1:]` are synonyms + trade names
- `OpenTargetsClient.get_drug()` still calls `get_all_drug_names` for cache warming only (result not stored on the model)
- `DRUG_QUERY` in `open_targets.py` trimmed to `id drugType` — name/synonyms/tradeNames dropped from the GraphQL request

