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

### Pipeline works best for focused-target drugs (2026-04-17)
- The pipeline is designed for drugs with a small number of specific molecular targets (e.g., semaglutide→GLP1R, metformin→MT-ND1/GPD2, imatinib→BCR-ABL). For these drugs, mechanism analysis meaningfully narrows the candidate hypothesis space and `shaped_associations` produces actionable hypothesis/contraindication classifications.
- **Known limitation — pleiotropic broad-spectrum drugs produce noisy output.** For drugs like aspirin (COX-1/COX-2 → systemic prostaglandin inhibition), NSAIDs, acetaminophen, corticosteroids, and broad-spectrum antibiotics, the target/mechanism is non-specific enough that (a) mechanism-sourced candidates span every organ system, (b) most literature searches find moderate-to-strong evidence regardless of candidate, and (c) `shaped_associations` falls back to mostly `neutral` or `confirms_known`. Output is not wrong, but the signal-to-noise is poor — the tool cannot distinguish real leads from "aspirin has been studied in everything."
- Users running the pipeline on pleiotropic drugs should expect noisy output and treat findings as exploratory rather than prioritized.

### FDA approval filter lives at two points in the supervisor (2026-04-17)
- Filter is applied in `supervisor_tools.find_candidates` (drops competitor diseases approved for the drug) and in `supervisor_tools.analyze_mechanism` (drops mechanism-surfaced diseases approved for the drug by mutating `MechanismOutput.shaped_associations` so approvals don't leak into the final `SupervisorOutput.mechanism` payload).
- Filter uses openFDA label text + LLM extraction via `get_fda_approved_diseases` → `extract_approved_from_labels`.
- `extract_fda_approvals.txt` prompt is intentionally conservative: matches only exact synonyms/renames (NASH = MASH = non-alcoholic steatohepatitis), **not** parent categories (NAFLD is not filtered when MASH is approved, because most NAFLD patients have simple steatosis without fibrosis and remain an open repurposing population).
- `MechanismOutput.summary` is LLM-generated pre-filter and may reference approved indications; when the FDA filter removes anything from the mechanism artifact, `output.summary` is blanked in `analyze_mechanism` to prevent approval leakage via narrative text. Structured fields are the source of truth.
- `fda_approval_check` cache key hashes `{label_texts, candidate_diseases}` but **not** the prompt text — prompt changes require manually clearing cache entries to take effect.

### ClinicalTrials.gov is Essie-backed and recall-first; analytical queries need MeSH post-filtering (2026-04-19)
- CT.gov's search engine (Essie) is tuned for patient-facing recall, not analytical precision: `query.cond=hypertension` returns trials for glaucoma, portal hypertension, pulmonary hypertension, and intracranial hypertension because Essie matches any trial mentioning "hypertension" in ConditionSearch metadata, eligibility criteria, or MeSH cross-references.
- Essie operator tuning (`AREA[Condition]EXPANSION[Term]`) was tested and rejected — it drops legitimate trials whose conditions are phrased slightly differently (e.g. "Diabetes Mellitus, Type 2" does not match `EXPANSION[Term]"type 2 diabetes"`).
- Solution is a two-stage pipeline: query CT.gov loosely (accept noise), then post-filter trials whose `derivedSection.conditionBrowseModule.meshes` or `ancestors` include the target MeSH D-number resolved from the indication string. Lives in `ClinicalTrialsClient._filter_by_mesh` and is wired through every indication-filtered method via the `target_mesh_id` kwarg.
- Indication → MeSH D-number resolution is a separate utility (`services.disease_helper.resolve_mesh_id`) that hits NCBI E-utilities. It is **independent** of the PubMed normalization path (`llm_normalize_disease` / `normalize_for_pubmed`) — the two do not chain.
- Trials with empty `meshes` AND empty `ancestors` are dropped (cannot be verified against the MeSH tree). Unresolvable indications cause the tool layer to short-circuit to an empty result. Both classes of silent loss are documented in `future.md`.
- Live regression baseline (date_before=2025-01-01, run 2026-04-19) — `tests/integration/data_sources/test_clinical_trials_mesh_filter.py`:
  - metformin × hypertension (D006973): 44 → 31
  - aspirin × diabetes mellitus (D003920): 50 → 43
  - semaglutide × hypertension (D006973): 5 → 2

