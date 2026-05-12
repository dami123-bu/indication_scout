# Findings

Each entry is one rule/fact + minimal why.

---

## Project Rules

### Never remove TODO comments
TODO comments stay during review/cleanup/refactor.

### Pydantic `coerce_nones` validator (2026-03-01)
Every external-data model needs `model_validator(mode="before")` to coerce `None` → default. Does not inherit to nested models.

### ChEMBL ID is the sole drug identifier (2026-04-16)
`DrugData` / `DrugProfile` carry only `chembl_id`. Names come from `get_all_drug_names(chembl_id)` (`[0]`=pref_name, `[1:]`=synonyms/trade). `DRUG_QUERY` GraphQL trimmed to `id drugType`.

### Pipeline works best for focused-target drugs (2026-04-17)
Specific targets (semaglutide, metformin, imatinib) yield signal. Pleiotropic drugs (aspirin, NSAIDs, steroids, broad-spectrum antibiotics) produce noise — every organ candidate looks supported. Treat output as exploratory for those.

### Semaglutide × NAFLD/MASH approval scope (2026-05-02)
Approved for MASH (narrow), NOT NAFLD (broad). Briefing showing MASH approved + trial-side `is_approved=False` for NAFLD is correct, not a bug.

### FDA approval filter lives at two points in the supervisor (2026-04-17)
Filter applied in `find_candidates` and `analyze_mechanism` (mutates `shaped_associations`). Uses openFDA + LLM. Prompt is conservative: matches exact synonyms/renames only, not parent categories (NAFLD not filtered by MASH). `MechanismOutput.summary` is blanked when anything is dropped. Cache key omits prompt text — clear cache after prompt edits.

### ClinicalTrials.gov is Essie-backed and recall-first; analytical queries need MeSH-precise filtering (2026-04-19, 2026-04-25)
CT.gov's Essie engine is recall-tuned, not precise. Use server-side `AREA[ConditionMeshTerm]"<MeSH preferred term>"` on both count and fetch calls; drug stays free-text `query.intr`. MeSH resolution via `services.disease_helper.resolve_mesh_id` (independent of PubMed normalization). Unresolvable indications short-circuit to empty.

### Clinical trials data layer rewritten to count + top-50 exemplars per pair (2026-04-25)
`ClinicalTrialsClient` exposes `search_trials` / `get_completed_trials` / `get_terminated_trials` / `get_landscape`, each returning (counts, top-50 by enrollment). Stop-category computed on read via `_classify_stop_reason`. `ApprovalCheck` artifact from `check_fda_approval` drives the `pair_completed` short-circuit.

### Agent prompt methodology — five-section template (2026-04-22)
Prompts follow TOOLS / SCHEMA / REPORTING / EMPTY RESULTS / INFERENCE / GROUNDING. Every INFERENCE rule must pass the audit test — if no tool returns the field it names, weaken/move/delete. Ban schema identifiers (`is_approved`, `pair_completed`, MeSH, etc.) in user-facing summaries.

### LangChain `content_and_artifact` does NOT expose the artifact to the LLM (2026-05-01)
Artifact stays Python-side; LLM only sees `content`. Anything the LLM must reason over goes in `content`. Per-trial detail now lives in `content` strings for `search_trials` / `get_completed` / `get_terminated` / `analyze_clinical_trials`. Formatters at `agents/_trial_formatting.py`; columns in `agent_data_contracts.md`.

### Literature agent's `expand_search_terms` tool was bypassing the production path (2026-05-10)
Production `svc.expand_search_terms(...)` was commented out, returning `[f"{drug} AND {disease}"]`. Restored. Lesson: thin tool wrappers are easy to dead-code silently.

### PubMed `AND` parser breaks on bare multi-word phrases (2026-05-10)
`drug AND multi word` returns 0. Use `<drug>[tiab] AND "<MeSH preferred term>"[MeSH]`. MeSH UID search (`D015431[MeSH]`) doesn't work — stick to preferred term.

### NCBI MeSH-db `esummary` returns empty records for valid UIDs (2026-05-10)
esummary broken for mesh db. Parse preferred term from esearch's `querytranslation` instead. Empty `idlist` is intermittent — 3-attempt retry. `MESH_RESOLVER_MAX_CONCURRENT = 5` caps parallelism.

### `expand_search_terms` now uses the resolved MeSH preferred term (2026-05-10)
`RetrievalService.expand_search_terms` resolves MeSH first, passes preferred term to LLM, falls back to raw on miss. Prompt instructs verbatim quoting of disease term. Output lowercase.

### `PubMedClient.search` now logs zero-hit queries and pre-call sleeps 1s (2026-05-10)
Zero-hit warning + 1s pre-call sleep inside semaphore.

### Reports do not include per-disease "FDA approval" yes/no (2026-05-10)
Field removed from summary line + report renderer. Approval relationship still surfaced inside blurb prose. Added "combination component" classification (e.g. bupropion/naltrexone → demote like "same disease").

### Query expansion is a wider net with a real older-literature trade (2026-05-10)
22-pair comparison: 1 rescue, 3 downgrades — synonym/mechanism queries surface older mechanism-era literature that synthesize honestly rates lower. Reviewers preferred expanded prose. Keep expansion default.

### Literature agent → supervisor flow: deterministic, no LLM rewrite of synthesize output (2026-05-10)
`finalize_analysis()` is now a pure termination signal. Supervisor-facing literature summary built deterministically: header + `evidence_summary.summary` verbatim + `key_findings` bullets. `LiteratureOutput.summary` removed. `strength == "none"` is overloaded — gates dropping on it alone may discard contraindication signals.

### Top-N evidence gate uses literature signal, not raw PMID count (2026-05-10)
`finalize_supervisor` drops when `n_trials == 0` AND (`strength == "none"` OR `study_count == 0`). PMID-count fallback only when synthesize didn't run. Strong/moderate/weak with 0 trials is kept.

### Candidate dedup centralized in supervisor `merge_and_dedup`, with hierarchical LLM pass (2026-05-11)
All cross-source dedup in one place. Mechanism agent buffers candidates; `merge_and_dedup` runs (1) EFO match, (2) name match, (3) OT resolve, (4) hierarchical LLM pass. MoA-aware survivor selection. On LLM error: keep everything (error by omission). `find_candidates_done` set in `finally`.

### Regression testing: golden SupervisorOutput + vcrpy cassettes (2026-05-11)
Compares structure + semantic overlap, not prose. `SupervisorOutput.model_dump_json` is the snapshot. vcrpy covers both `aiohttp` clients and `httpx` Anthropic SDK. Modes via `SCOUT_CASSETTE_MODE`: `replay` / `record` / `live`. Jaccard thresholds in `regression/constants.py`. DB is live at replay time.

### Older trial readouts can be missed by literature retrieval (2026-05-11)
Pre-2018 trial publications often lack NCT-in-abstract tagging. Concrete misses: baricitinib × psoriasis (Papp 2016, NCT01490632) and bupropion × ADHD (Wilens 2005, NCT00048360). CT.gov `references` link absent, NCT-tag PubMed search returns 0, semantic retrieval ranks paper past `PUBMED_MAX_RESULTS=100` under default recency sort. Cheapest fix not yet deployed: switch PubMed `sort=relevance`. Treat "no published readout" framing as possibly an artifact.

### Clinical-trial filter pipeline removed; counts come through from CT.gov untouched (2026-05-11)
Stripped top-2 MeSH / LLM primacy / drug-alias filters. Total_count and by_status flow through unmodified — eliminates extrapolation and cross-count inconsistency. Holdout date-scrubber preserved. Trade-off: trial table may include off-primary-MeSH and eligibility-only-drug trials; LLM judges from titles + MeSH. Helpers left in place per CLAUDE.md.

### Hierarchical candidate dedup disabled; was collapsing actionable subtypes (2026-05-11)
LLM dedup pass collapsed metformin's PCOS / hepatic steatosis / gestational diabetes into "metabolic disease" parent — PCOS (#1 off-label) disappeared. Commented out at `supervisor_tools.py:470-528`. Exact-match steps still run. Demotion machinery (`broader_overlapping` etc.) still handles parent-of-approved cases in `finalize_supervisor`. Future: curated `DEDUP_GROUPS` table, not LLM judgment.

### Supervisor candidate-investigation is now deterministic (2026-05-11)
Was: LLM picked 3–5 of merged list. Two bupropion runs 5h apart differed by ~40%. Now: investigate ALL candidates in order, cap 6. Cost: 30–40% more agent calls. Bupropion × substance dependence demotion still buries actionable methamphetamine signal — curated `DEDUP_GROUPS` override pending.
