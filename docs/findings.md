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

### ClinicalTrials.gov is Essie-backed and recall-first; analytical queries need MeSH-precise filtering (2026-04-19, updated 2026-04-25)
- CT.gov's search engine (Essie) is tuned for patient-facing recall, not analytical precision: `query.cond=hypertension` returns trials for glaucoma, portal hypertension, pulmonary hypertension, and intracranial hypertension because Essie matches any trial mentioning "hypertension" in ConditionSearch metadata, eligibility criteria, or MeSH cross-references.
- Essie operator tuning (`AREA[Condition]EXPANSION[Term]`) was tested and rejected — it drops legitimate trials whose conditions are phrased slightly differently (e.g. "Diabetes Mellitus, Type 2" does not match `EXPANSION[Term]"type 2 diabetes"`).
- **Original approach (now superseded):** a two-stage pipeline that queried CT.gov loosely and post-filtered client-side via `ClinicalTrialsClient._filter_by_mesh` against `derivedSection.conditionBrowseModule.meshes` / `ancestors`. This produced a "shown vs total" mismatch — the count call was free-text but the displayed exemplars were post-filtered, so the two numbers measured different populations.
- **Current approach (2026-04-25):** server-side MeSH-precise filtering via CT.gov's `AREA[ConditionMeshTerm]"<MeSH preferred term>"` syntax, applied to **both** the count call and the fetch call. The same query is used for counts and exemplars, so `len(trials) <= total_count` always and the only mismatch is the 50-record `CLINICAL_TRIALS_FETCH_MAX` cap. Drug side stays free-text via `query.intr` so trials whose intervention isn't MeSH-tagged are still caught. See `trial_refactor.md` for the migration plan.
- Indication → MeSH descriptor resolution is a separate utility (`services.disease_helper.resolve_mesh_id`) that hits NCBI E-utilities. It returns `(descriptor_id, preferred_term)` (e.g. `("D003866", "Depressive Disorder")`). The `preferred_term` is what feeds CT.gov's `AREA[ConditionMeshTerm]`. It is **independent** of the PubMed normalization path (`llm_normalize_disease` / `normalize_for_pubmed`) — the two do not chain.
- Unresolvable indications cause the tool layer to short-circuit to an empty result (search/completed/terminated/landscape all return their empty-result shape with a warning log). The "trials with empty meshes/ancestors are dropped" failure mode no longer applies because filtering is server-side; the silent-loss surface is now confined to MeSH resolver failures (see `future.md`).
- Historical regression baseline (client-side post-filter, date_before=2025-01-01, run 2026-04-19) — `tests/integration/data_sources/test_clinical_trials_mesh_filter.py`:
  - metformin × hypertension (D006973): 44 → 31
  - aspirin × diabetes mellitus (D003920): 50 → 43
  - semaglutide × hypertension (D006973): 5 → 2

### Clinical trials data layer rewritten to count + top-50 exemplars per pair (2026-04-25)
- `ClinicalTrialsClient` now exposes three pair-scoped query methods that all share the same shape: `(counts, top-50 trials sorted by enrollment desc)`. Counts come from cheap `countTotal=true&pageSize=1` calls; exemplars come from a single paginated fetch capped at `CLINICAL_TRIALS_FETCH_MAX = 50`.
  - `search_trials(drug, mesh_term, date_before)` → `SearchTrialsResult(total_count, by_status={RECRUITING, ACTIVE_NOT_RECRUITING, WITHDRAWN}, trials)`. TERMINATED and COMPLETED counts are deliberately not surfaced here — they live on their dedicated tools to avoid double-counting.
  - `get_completed_trials(drug, mesh_term, date_before)` → `CompletedTrialsResult(total_count, phase3_count, trials)`. Phase 3 is the only phase the supervisor's summary cites.
  - `get_terminated_trials(drug, mesh_term, date_before)` → `TerminatedTrialsResult(total_count, trials)`. Each Trial carries `why_stopped`; stop-category classification (`safety` / `efficacy` / `business` / `enrollment` / `other` / `unknown`) is computed on read at the tool layer via `_classify_stop_reason`. There is no pre-computed `stop_category` field on the model.
  - `get_landscape(mesh_term, date_before, top_n)` → `IndicationLandscape(total_trial_count, competitors, phase_distribution, recent_starts)`. Server-side Drug/Biologic phase filter; vaccines excluded by name keyword.
- The deleted models (`TrialOutcomes`, `TerminatedTrial`, `WhitespaceResult`, `IndicationDrug`) and the old `detect_whitespace` tool are gone. Whitespace is now derivable as `SearchTrialsResult.total_count == 0`.
- `ApprovalCheck` is a new artifact returned by the `check_fda_approval` tool: `is_approved`, `label_found`, `matched_indication`, `drug_names_checked`. The clinical-trials agent must call `check_fda_approval` whenever `get_completed` reports any trials — it is the only tool that can tell the agent whether a completed trial led to approval (the `pair_completed` short-circuit). When `is_approved=True`, the prompt forces a one-sentence summary and skips the rest of the report.
- The clinical-trials sub-agent's `ClinicalTrialsOutput` shape is `search / completed / terminated / landscape / approval / summary`. The supervisor's `OUTCOME ACCOUNTING` prompt section still references the older `drug_wide / indication_wide / pair_specific / pair_completed` scope vocabulary — those scopes were dropped from `TrialOutcomes` in this refactor, and the supervisor prompt has not yet been updated to match.

### Agent prompt methodology — five-section template (2026-04-22)
- All agent system prompts should follow the structure documented in `prompt_plan.md`: TOOLS / SCHEMA / REPORTING / EMPTY RESULTS / INFERENCE / GROUNDING. Mixing schema facts, reasoning heuristics, and style rules in one block hides tool/prompt mismatches.
- Every INFERENCE rule must pass the **audit test**: "if I delete every tool's output except the exact fields this rule names, can it still be applied, and does it correctly state what it CAN'T conclude?" If not, weaken / re-route through a new tool / move / delete.
- "Unless X" clauses where no tool returns X are the canonical bug shape. The clinical_trials prompt used to say "treat a completed Phase 3 as closed unless there is direct evidence the readout was positive" — no tool returned readout evidence, so every completed Phase 3 collapsed to "failed." This is what missed semaglutide × NASH (FDA-approved for MASH Aug 2025).
- Fix for clinical_trials: added `check_fda_approval` tool (`ApprovalCheck` artifact) that wraps `services.approval_check.get_fda_approved_diseases`, routed the `pair_completed` inference rule through it. When `is_approved=True`, a short-circuit rule forces a one-sentence summary and skips the rest of the report — approved drugs are not repurposing candidates.
- User-facing summaries must not leak internal field or tool names. Models parrot schema identifiers (`pair_completed`, `is_approved`, `drug_names_checked`, `check_fda_approval`, `stop_category`, `is_whitespace`, MeSH, etc.) unless the REPORTING section explicitly bans them with plain-English translations.

