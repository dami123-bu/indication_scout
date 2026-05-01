# Future Improvements

## NCBI rate limiting — proper semaphore + spacing gate (added 2026-04-26)

Symptom (observed 2026-04-26 on dasatinib × breast / melanoma run): three
concurrent supervisor pairs hit `resolve_mesh_id` simultaneously, all 429'd on
the same MeSH descriptor URL, and all entered a 90s reactive backoff in
lockstep. Trace showed the same warning line three times for the same
`indication='melanoma'` and the same descriptor id, firing in the same
scheduler tick.

Current mitigation (in place): pre-emptive jittered `asyncio.sleep` before
every `_ncbi_get_json` attempt (`_NCBI_PRECALL_SLEEP_BASE=0.5s`,
`_NCBI_PRECALL_SLEEP_JITTER=0.4s`). Crude but cheap. It desynchronizes
concurrent callers within ~0.9s of each other but does not enforce a global
rate ceiling — under heavier supervisor parallelism (e.g. 5+ candidates) the
429s will return.

Proper fix when this resurfaces:

1. **Module-level `asyncio.Semaphore(8)`** in `disease_helper.py` — one slot
   below NCBI's 10 req/s ceiling with API key. Every NCBI call acquires it.
2. **Module-level `asyncio.Lock` + last-call timestamp** for minimum spacing.
   Under the lock: wait until `now - last_call >= 0.12s`, update `last_call`,
   release. This guarantees spacing regardless of caller count.
3. Wrap both around `_ncbi_get_json` so all NCBI traffic flows through the
   same gate (esearch + esummary, MeSH resolver and any future caller).
4. Drop the pre-emptive `asyncio.sleep` once the semaphore is in place — the
   semaphore is strictly better (slow only when traffic is heavy).
5. Drop the inline `asyncio.sleep(0.1)` calls in `resolve_mesh_id` (lines 347,
   361, 388) for the same reason — they're a per-callsite version of the same
   idea, but they don't coordinate across coroutines.

Also worth doing alongside:
- **Cache negative MeSH results.** The current docstring of `resolve_mesh_id`
  notes "Does NOT cache None results." That means a missing or unresolvable
  descriptor (e.g. a typo, a rare disease) re-queries NCBI on every call from
  every pair. With the semaphore in place, this becomes the dominant remaining
  source of avoidable load.

Estimated effort: low (single file, ~30 lines). Defer until the pre-emptive
sleep proves insufficient under realistic supervisor parallelism.

---

## Supervisor — drug-level shared store (added 2026-04-26)

Observed in a semaglutide snapshot: the supervisor demoted NAFLD as "settled
and unfavorable" because `(semaglutide, NAFLD)` returned `is_approved=False,
completed.phase3=1`. Locally correct, globally wrong — semaglutide IS approved
for NASH (the inflammatory subset of NAFLD, per 2024 Wegovy expansion). The
supervisor never saw that fact because no candidate was named "NASH."

Two structural gaps:

1. **No drug-level facts visible to the supervisor.** Each sub-agent call is
   per-pair. FDA-approval discoveries, ChEMBL aliases, and mechanism context
   get computed and discarded once the pair is analyzed. A fact discovered for
   `(drug, candidate_A)` is invisible when the supervisor reasons about
   `(drug, candidate_B)`.
2. **Strict GROUNDING RULE blocks training-knowledge inference.** Even if the
   model knows NASH ⊂ NAFLD, the prompt blocks that inference (rightly —
   opening that door invites hallucinated approvals).

Fix: a closure-scoped shared store inside `build_supervisor_tools`, populated
by sub-agents as side effect, surfaced to the supervisor via two tools.

**Store fields (Tier 1, in scope):**
- `drug_aliases` (ChEMBL trade/generic names)
- `approved_indications` (`(text, matched_label_text)` tuples)
- `mechanism_targets` (gene + action_type pairs)
- `mechanism_disease_associations` (high-score target→disease pairs)

**Tools (Tier 2):**
- `analyze_drug(drug_name)` — runs once at start, resolves ChEMBL aliases +
  FDA approval check across known indications. Writes to store. Returns
  briefing as content.
- `get_drug_briefing()` — read-only view of current store, rendered as
  markdown. Supervisor calls before `finalize_supervisor`.

**Sub-agent write-throughs:**
- `analyze_mechanism` populates `mechanism_targets` and
  `mechanism_disease_associations`.
- `analyze_clinical_trials` appends any `is_approved=True` matched_indication.

**Briefing shape (terse, structured, no prose):**
```
DRUG INTAKE: semaglutide
- Trade names: Ozempic, Wegovy, Rybelsus
- FDA-approved indications:
  - Type 2 diabetes mellitus
  - Chronic weight management
  - MASH
- Targets: SLC6A2 (INHIBITOR), SLC6A3 (INHIBITOR)
- Top mechanism-disease associations:
  - SLC6A2 → ADHD (score 0.95)
  - SLC6A3 → ADHD (score 0.93)
```

**Prompt change:** add to RECONCILIATION RULE that the supervisor must call
`get_drug_briefing()` before finalize and check whether any candidate is
related to an approved indication (subset/superset/sibling). If so, name the
relationship explicitly — do not treat the candidate as a closed unfavorable
hypothesis when its sister indication is already approved.

**Implementation order:** closure dict → `analyze_drug` tool → briefing
renderer → `get_drug_briefing` tool → mechanism write-through → trials
write-through → supervisor prompt update → regression test against
semaglutide × NAFLD case.

**Out of scope (added back when concrete trigger conditions appear):**
- Drug-level safety signals (accumulate from terminated `why_stopped`).
  Add when the same drug repeatedly stops for the same reason across pairs.
- `related_terms` for approved indications (precomputed synonym/subset/
  superset hints). Add if the LLM proves unable to make the NASH↔NAFLD
  inference from the flat approved list alone.

---

## Reconsider drug_wide and indication_wide scopes (added 2026-04-25)

The current refactor (see `trial_refactor.md`) drops `drug_wide` and
`indication_wide` from `TrialOutcomes` entirely. They were causing the same
unrelated trials (e.g. a bupropion weight-loss futility stop) to appear under
every candidate disease in the report, where they have no causal relation to
the candidate.

Both signals are legitimately useful at the right layer — they were just being
computed and reported at the wrong one (per pair, repeated per candidate).
When time allows, consider re-introducing them deliberately:

- **drug_wide** (this drug × any indication, safety/efficacy terminations) is a
  drug-level fact. Natural home: the supervisor or mechanism agent, fetched
  once per drug and surfaced once at the top of the report — not per candidate.
- **indication_wide** (any drug × this indication, terminations) is an
  indication-level fact about how hard the disease area is. Natural home:
  inside `get_landscape`, alongside the competitor list — that tool already
  owns the indication-level view.

See also the existing "Clinical Trials agent — collapse to a single precise
search" section near the bottom of this file, which framed the same direction.

---

## Clinical Trials Query Quality

### 0. MeSH resolver — ambiguous term handling

**!! HIGH PRIORITY — COMMON LAY TERMS SILENTLY ZERO-OUT THE TOOL LAYER !!**

CONFIRMED 2026-04-19 VIA `tests/integration/data_sources/test_clinical_trials_mesh_filter.py`:
THE BARE TERM `"diabetes"` RESOLVES TO `None` BECAUSE THE MESH PREFERRED
DESCRIPTOR IS `"Diabetes Mellitus"` (D003920). EVERY OTHER COMMON ONE-WORD
INDICATION WHOSE MESH PREFERRED FORM HAS A QUALIFIER (e.g. `"asthma"` vs.
`"Asthma, Exercise-Induced"`, `"hypertension"` worked but `"depression"` will
hit the symptom D003863 instead of the disorder D003866) IS LIABLE TO THE
SAME FAILURE. WHEN THE RESOLVER RETURNS `None`, THE TOOL LAYER RETURNS THE
EMPTY-RESULT SHAPE — NO TRIALS, NO ERROR, NO USER-VISIBLE WARNING. AGENTS
WILL TREAT THIS AS GENUINE WHITESPACE.

The basic `resolve_mesh_id` returns the first NCBI esearch hit. This will pick the
wrong descriptor for genuinely ambiguous terms (e.g. `"depression"` → `D003863`
symptom instead of `D003866` disorder). Add, in this order, only if the basic
resolver proves insufficient:
- Broader-name fallback when `[MeSH Terms]` returns zero hits.
- Confidence check on the top hit (edit-distance or LLM) to detect mismatches.
- LLM fallback via `query_small_llm` + a new `prompts/resolve_mesh.txt` for
  disambiguation, biased toward the disorder form for repurposing context.

### 1. Expand drug synonyms before querying
ChEMBL/DrugBank already have synonym lists. Pass the top N synonyms as an OR query
(e.g. `metformin OR glucophage OR dimethylbiguanide`) to `query.intr`.
This addresses false whitespace signals caused by trials registered under brand names
or salt forms rather than the INN.

### 2. Phase 3
Put in something about pivotal trials in the final synthesis
A pivotal study is a clinical trial whose results form the primary basis for a regulatory approval decision (FDA, EMA, etc.).

Key characteristics:

Phase 3 (occasionally Phase 2 in rare disease / oncology / accelerated approval contexts)
Adequate and well-controlled — randomized, blinded, with a pre-specified primary endpoint
Powered to demonstrate efficacy on that endpoint with statistical significance
Conducted under an IND with the design pre-agreed with the regulator (often via End-of-Phase-2 meeting / SPA)
The FDA typically requires two independent pivotal trials ("substantial evidence of effectiveness", FDCA §505(d)), though one pivotal + confirmatory evidence is accepted in some cases (e.g. rare diseases, oncology with strong effect size).

Example: For dupilumab in eosinophilic esophagitis, the pivotal trial was LIBERTY EoE TREET (NCT03633617) — a Phase 3 randomized placebo-controlled trial whose Part A and Part B results supported the May 2022 approval.

Distinct from:

Supportive studies — Phase 2 dose-finding, PK/PD, mechanism studies cited alongside but not the basis for approval
Post-marketing studies (Phase 4) — run after approval to confirm safety/effectiveness in broader populations

## Literature Agent — Adaptive Search

The MVP uses a fixed call sequence with no retry logic. Once time allows, add:

- **Low hit count broadening**: if `fetch_and_cache` returns fewer than ~20 PMIDs, call
  `expand_search_terms` again with a broader disease term (e.g. `"non-alcoholic steatohepatitis"`
  → `"liver disease"`) before proceeding to `semantic_search`.
- **Low similarity retry**: if `semantic_search` returns all similarity scores below ~0.6,
  try `fetch_and_cache` with different queries before calling `synthesize`.

---

## Clinical Trials — WITHDRAWN Coverage Gap

`get_terminated` only queries `filter.overallStatus=TERMINATED`. Trials with status
`WITHDRAWN` (pulled before any participants were enrolled) are not captured at all.

These are a distinct signal: a sponsor committing to a trial and then backing out before
enrollment may indicate early safety signals, failed IND, or strategic retreat — all
relevant to repurposing analysis.

Consider adding a separate `WITHDRAWN` query (drug-wide and/or indication-specific) and
a corresponding `WithdrawnTrial` model or an extended `TerminatedTrial` with a
`pre_enrollment: bool` flag to distinguish the two statuses.

---

## WhitespaceResult Schema Gap

`is_whitespace` is binary, but that misses a third state: "early stage, unproven."

Example: metformin + glioblastoma returns `is_whitespace=False` with `exact_match_count=9`,
but those 9 trials are all Phase 1/2 with small enrollment. The agent correctly identifies
"not whitespace, but not competitive either" — but that nuance lives only in the free-text
summary, not in the structured data.

The phase/maturity dimension is not captured in `WhitespaceResult`. Consider adding:
- A `max_phase` field (highest phase among exact-match trials)
- A `maturity` enum: `whitespace` / `early_stage` / `established`
- Or aggregate enrollment of exact-match trials as a proxy for how well-covered the space is

---

## Disease Name Canonicalisation

The supervisor may investigate mechanism-surfaced diseases under names that differ from their find_candidates equivalents (e.g. "non-alcoholic steatohepatitis" vs "fatty liver disease"). Both are valid but the report does not collapse them. Cross-reference against Open Targets disease IDs if exact matching is needed.

---

## Clinical Trials — Trials Dropped by MeSH Post-Filter

Once the MeSH-resolver + post-filter refactor lands (see `plan_trial_refactor.md`), three classes
of trials (or entire queries) will be silently dropped:

1. **Trials with empty `conditionBrowseModule.meshes`.** CT.gov's MTI auto-tagger occasionally
   produces no MeSH tags for a trial — we saw this for NCT06972992 ("Chronic Weight Management")
   and NCT06354101 ("Metabolic Health"). With no MeSH terms, they cannot match a target MeSH ID
   and get dropped even if their free-text condition is a legitimate match.
2. **Trials whose MeSH tags are siblings, not descendants, of the target.** Example:
   "Depression" (D003863, a symptom) vs "Depressive Disorder" (D003866, the disease) are
   different MeSH branches. A resolver that picks D003863 will drop all MDD trials tagged
   D003866. Mitigated by a smarter resolver, but residual siblings-not-descendants edge cases
   will remain.
3. **Unresolvable indications drop the entire query.** If the MeSH resolver returns `None` for
   an indication string (neither NCBI nor the LLM fallback produces a confident ID), the whole
   trial search short-circuits to an empty result. No string-based fallback runs. This is a
   deliberate choice for correctness — better no result than a noisy one — but it means rare
   diseases, novel indication phrasings, or typos will return nothing rather than a best-effort
   string match. Consider surfacing these failures to the agent layer so the orchestrator can
   retry with a normalized/synonymous indication before giving up.

Consider a fallback bucket: keep these trials in a separate `unfiltered` list on the result
so agents can decide whether to inspect them manually, rather than losing them entirely.

---

## Report — `pair_completed` Rendering

`format_report._fmt_clinical_trials` currently skips `stop_category` for the
`pair_completed` bucket because `TrialOutcomes.pair_completed` is `list[Trial]`,
and `Trial` has no `stop_category` field (only the three `TerminatedTrial`
buckets do). Completed trials are rendered with phase + `overall_status` only.

This loses signal: the interesting question for a completed pair-specific trial
is whether it *met its primary endpoint*. ClinicalTrials.gov marks endpoint
failures as COMPLETED, not TERMINATED, so the report currently cannot
distinguish "ran and succeeded" from "ran and missed endpoint."

Options when time allows:
- Surface `primary_outcomes` text in the rendered line for `pair_completed`.
- Add an `outcome_category` (e.g. met / missed / unclear) to `Trial` or a new
  `CompletedTrial` model, populated from results-section parsing or LLM
  classification of the primary-outcomes text.
- Pull results data via the CT.gov results endpoint for these specific NCT IDs.

## FDA approval check — fallback when no label is found (added 2026-04-22)

`check_fda_approval` currently relies on `FDAClient.get_all_label_indications`,
which queries openFDA's SPL (Structured Product Labeling) dataset. When a drug
is withdrawn from the US market, its label is pulled from active distribution
and openFDA returns zero labels — even if the FDA originally granted approval.
Aducanumab (Aduhelm) is the canonical case: accelerated approval in June 2021
for Alzheimer's disease, voluntarily withdrawn by Biogen in January 2024, now
returns `label_found=False` from our tool.

We handle this correctly in the prompt (the `is_approved=false AND
label_found=false` INFERENCE rule forces the agent to report approval status
as UNKNOWN and not treat it as negative evidence), but UNKNOWN is strictly
less useful than the truth. A withdrawn-but-originally-approved drug is
*still not a repurposing opportunity* in the conventional sense — it passed
regulatory review once, even if it's no longer sold.

Fallback options, in rough order of effort:
- Query openFDA's drugsfda endpoint instead of (or in addition to) SPL —
  drugsfda carries approval history and retains withdrawn products with a
  status marker. A hit there with status "Discontinued" or similar would let
  us distinguish "withdrawn after approval" from "never approved."
- Query the FDA orange/purple book datasets directly for historical approval
  records.
- LLM fallback on a curated drug name list: when SPL returns nothing, ask a
  small LLM whether the drug was ever FDA-approved and for what indication.
  Cheaper than a new data source but less auditable.
- At minimum, add a third `approval_history_unknown` state (or a separate
  `ever_approved: bool | None` field) so the agent can distinguish "drug has
  current label, indication not on it" from "drug has no current label,
  historical approval unchecked."

## Mechanism — conflicting action types on a single target (added 2026-04-22)

When a drug has two MoAs on the same target with opposing directions (one
`INHIBITOR`, one `AGONIST`), `build_symbol_action_types` in
`mechanism_shapes.py` records both in the `action_types` set for that
symbol. `classify_association_shape` then sets `is_inhibitor` AND
`is_activator` both to True, and whichever `if` branch runs first wins:

```
if (is_inhibitor and disease_gof) or (is_activator and disease_lof):
    shape = "hypothesis"
elif (is_inhibitor and disease_lof) or (is_activator and disease_gof):
    shape = "contraindication"
```

For a GOF disease this always returns `hypothesis`; for a LOF disease
always `contraindication`. The conflicting direction on the same target is
silently dropped. Low priority — rare in the Open Targets data we've
seen — but worth either:
- Adding a comment on `classify_association_shape` flagging the tie-break
  behaviour explicitly.
- Returning a dedicated `ambiguous_direction` shape when
  `is_inhibitor AND is_activator`, so downstream consumers know the drug's
  direction on that target is itself contested.
- Carrying the set of action_types into the rationale string so the
  conflict is at least visible in the audit trail.

## FDA approval check — formulation-relevance ranking hides indications (added 2026-04-25)

`get_fda_approved_disease_mapping` queries openFDA per ChEMBL alias with
`OPENFDA_LABEL_LIMIT=5`. For drugs sold under multiple formulations with
different indication profiles, openFDA's relevance ranker tends to return
labels for the most populous formulation first, pushing the labels with
rarer indications past the cutoff.

Bupropion is the canonical case (confirmed 2026-04-25 via probe):
- Total openFDA labels for bupropion: 255
- Smoking-cessation indication lives only on the SR formulation (~45 labels)
- MDD / SAD indications live on XL and immediate-release formulations (~190 labels)
- Top-5 by openFDA's ranker returns 5 XL/IR labels — none mention smoking
- The brand name "Zyban" was discontinued and is no longer in openFDA at all,
  so the alias-expansion fallback can't surface the SR label by trade name

Same pattern hits:
- fluoxetine / PMDD (Sarafem-era label not in top-5)
- doxycycline / Lyme disease (Vibramycin/Acticlate not in top-5)
- acetaminophen / osteoarthritis (Tylenol Arthritis not in top-5)
- amoxicillin / pneumonia, ondansetron / nausea, ivermectin / scabies, etc.

We patch each via CURATED_FDA_APPROVED_CANDIDATES, but the curated list
keeps growing.

Architectural fixes, in rough order of effort:
- **Increase per-alias `OPENFDA_LABEL_LIMIT` to 100, then dedup by
  `indications_and_usage` text content.** Most of the manufacturer-multiplicity
  duplicates (e.g. ~190 "Bupropion Hydrochloride" XL labels with identical
  indication strings) collapse to a handful of unique indication strings.
  Smoking-cessation would survive the dedup. Tradeoffs: bigger HTTP payloads,
  more LLM context, slower. Earlier global bump to limit=25 caused regressions
  on atorvastatin (prompt got noisy). A higher limit + content-dedup before
  prompting may behave better.
- **Per-candidate filtered query.** When checking `(bupropion, smoking cessation)`,
  hit openFDA with a search filter that requires the candidate term in
  `indications_and_usage` — surfaces the relevant labels deterministically.
  Costs one openFDA call per (drug, candidate) pair instead of per drug.
  Architectural change to the function signature / call pattern.
- **Filter by `dosage_form` or `route`** — bias the per-alias query toward
  SR vs XL. Drug-specific knowledge required; doesn't generalize.

## FDA approval check — wrong primary endpoint (added 2026-04-25)

The bigger architectural point behind the bupropion / fluoxetine / doxycycline
/ acetaminophen pattern: we're using `/drug/label.json`, which is the wrong
endpoint for "what is this drug approved for?". The label endpoint serves
*Structured Product Labels* — manufacturer marketing/prescribing documents,
filed once per (manufacturer × formulation × strength × packaging). Approvals
get repeated 50-200 times across SPLs, indications get scattered across
formulation-specific labels, and the ranker decides which slice we see.

The FDA *does* maintain clean approval data; we're just reading it out of
the wrong source. Better-aligned data sources:

- **openFDA `/drug/drugsfda.json`** — keyed by FDA application number
  (NDA/BLA/ANDA). One record per *application*, with submission history,
  approval letters, and indication strings. No SPL multiplicity, no
  formulation-relevance ranking, no per-manufacturer duplication. This is
  structurally what we want — "approval" is an application-level event,
  not a label-level event. Worth a prototype to see how clean the
  indication strings actually are in this dataset.
- **FDA Orange Book / Purple Book** — official drug-approval lists,
  downloadable as monthly text files. Per-application records with
  active ingredient, dosage form, route, approval date. No API but stable
  and bulk-loadable. Authoritative source.
- **DailyMed** — NLM's frontend to the same SPLs, but with structured
  indication metadata and RxCUI / NDC cross-references. Same underlying
  data as openFDA labels, better-organized retrieval.
- **RxNorm + NDF-RT** — pre-computed "may treat" / "is treated by"
  relationships keyed by RxCUI. Skips the need to extract indications
  from text entirely.

Estimated impact of switching to drugsfda.json: ~60-70% of current curated
entries would be unnecessary (the ones that are real approvals just buried
in SPL ranking noise). Genuine OT/openFDA gaps and LLM judgment errors
would persist regardless of source. Estimated effort: medium — new client
or new endpoint method on FDAClient, new prompt that consumes structured
indication strings instead of free-text label sections, sanity-test against
v1/v2/v3/v4 probes before swapping the production code path.

## Clinical Trials count truncation — page cap

`data_sources/clinical_trials.py::_count_trials` is hard-capped at 10 pages of
ClinicalTrials.gov results. When a query exceeds that (broad indications like
"nonalcoholic steatohepatitis", "depression", "renal insufficiency"), the
function stops early and logs:

    _count_trials: page cap (10) hit for drug=None indication=<X> mesh_id=<Y>;
    returned count is a lower bound

The returned count is therefore a **lower bound**, not the true total. For most
decisions ("is there meaningful trial activity?") the lower bound is enough,
but downstream consumers should be aware:
- "trial landscape size" for popular indications is truncated.
- Comparisons of trial counts across indications can be misleading when both
  are above the cap.

Same caveat applies to `get_landscape` — it logs a separate "pre-filter fetch
saturated at 20 pages" warning for the broader landscape fetch.

Options if precision matters for a specific path:
- Lift the cap for that call site (slow, more API calls).
- Switch to ClinicalTrials.gov's `countTotal` endpoint if/when available, so we
  get the exact count without paging.
- Cache the lower-bound counts and only re-paginate when an analysis explicitly
  needs an exact number.

Estimated effort: low for path-specific cap lifts; medium for a clean
"countOnly" mode that bypasses pagination.

## Clinical Trials agent — collapse to a single precise search

The clinical trials agent currently issues four scoped queries per (drug,
indication) pair via `data_sources/clinical_trials.py::get_terminated`:

  - drug_wide: this drug, any indication, terminated only.
  - indication_wide: this indication, any drug, terminated only.
  - pair_specific: this drug + this indication, terminated.
  - pair_completed: this drug + this indication, completed.

The LLM then has to reason across all four scopes simultaneously. Two issues:

1. The agent's output frequently surfaces counts ("12 completed trials")
   without making clear which scope they came from, and downstream summaries
   blur the boundary between drug-scoped, indication-scoped, and pair-scoped
   findings.
2. The MeSH descriptor used for `_filter_by_mesh` rolls subtypes up into the
   parent (e.g. NASH/MASH trials are tagged with the NAFLD MeSH descriptor
   D065626), so even pair-specific counts can lump clinically distinct
   populations together. Combined with the multi-scope structure, the user
   has no easy way to tell what the count represents.

Proposed direction: drop the multi-scope structure entirely and do one
precise pair-scoped search per call. Report what's there; don't aggregate
indication-wide attrition or drug-wide failures into the same answer.
Whitespace / landscape / drug-wide failure context becomes a separate tool
call (or moves out of the clinical-trials agent entirely), so each tool
returns one clean signal.

Out of scope for now — the mechanism agent is unaffected, and the current
behavior is acceptable for it. This change should be a deliberate redesign
of the clinical trials agent's tool surface, not a quick patch.

Estimated effort: medium — touches the data-source method signature, the
agent's tool definitions, the agent's prompt, and downstream supervisor
prompts that reason about completed/terminated counts.

## Clinical Trials sub-agent — two-stage list → drill-down (added 2026-04-30)

The clinical_trials sub-agent's tools use `@tool(response_format="content_and_artifact")`.
LangChain serializes only `ToolMessage.content` into the API payload sent to the LLM; the
`artifact` (the typed Pydantic object carrying the trials list, phase, why_stopped, etc.)
stays Python-side and is invisible to the model. Today the content strings carry only
aggregate counts ("4 total"), so the sub-agent's LLM has no per-trial visibility — yet the
system prompt explicitly tells it to "look at the phase field on each Trial in the returned
list" (clinical_trials_agent.py:45-46) and to "inspect search_trials.trials for any UNKNOWN
entries with Phase 3" (clinical_trials_agent.py:69-72). Symptom: the sub-agent's prose
contradicts the report's own trials table — claims "no Phase 3" while NCT00763867 (RELAX)
and NCT01726049 are listed as Phase 3 right below it.

A near-term content-string enrichment (per-trial NCT id + phase + title in the content
returned by `search_trials` / `get_completed` / `get_terminated`) fixes the immediate bug
and is what we'd ship first. This entry is about the longer-arc architectural shape.

**Proposed direction: two-stage list → drill-down.**

- Tools return a lightweight per-trial row by default — NCT id, phase, status, title, and
  (for terminated) classified stop reason — enough to triage but not enough to bury the
  prompt. The artifact still carries the full Trial objects for downstream Python.
- Add a `get_trial_details(nct_id)` tool the sub-agent can call to fetch the full Trial
  fields (brief_summary, primary_outcomes, enrollment, dates, references) for a specific
  trial it has flagged as worth a closer look.
- Update the system prompt to describe the triage-then-drill pattern explicitly so the LLM
  doesn't try to drill on every trial.

**Why this shape and not "dump everything to the LLM":**

Returning fully serialized Trial JSON for every result is the seductive-but-wrong shape.
With ≤50 trials × full Trial schema (title, brief_summary, mesh_conditions, mesh_ancestors,
interventions, primary_outcomes, references) per tool call × multiple tool calls per
candidate × ~15 candidates per run, the token cost is heavy (estimate: 30-80KB per tool
call before any trimming). Worse, attention quality degrades over giant JSON dumps — more
data does not equal better judgment past a point. The two-stage shape matches how a human
investigator actually works: scan the list, pick the few interesting ones, dig into those.

**Why not just stop at the curated content string forever:**

The curated-content fix is fine for the fields we know matter (phase, title, why_stopped).
But the LLM cannot reason over fields we did not pre-select. Example failures the
content-string fix cannot catch: a `brief_summary` that mentions a narrower subtype the
title omits; a `primary_outcomes` text that hints at endpoint structure; an `enrollment`
number that signals whether a Phase 2 was adequately powered. Two-stage drill-down lets
the LLM ask for that data only when it matters, without paying the token cost on the 90%
of trials where it doesn't.

**Costs:**

- More tool calls per run → more agent loop iterations → more LLM turns. Latency goes up.
  Mitigate by capping drill-downs per analysis (e.g. ≤3 per pair) in the prompt.
- Sub-agent prompt grows to describe the new tool and the triage pattern.
- Need a new `get_trial_details` data-source method or a way to fetch a single NCT id from
  the existing CT.gov client. The full Trial is already inside the artifact for trials in
  the top-50 — drill-down only needs to go to the network for trials past the cap.

**Risk:**

Medium. Touches the sub-agent prompt, adds a new tool, and changes the agent's per-run
turn count. Test coverage should include a regression scenario where the LLM correctly
drills on a flagged trial (e.g. an UNKNOWN-status Phase 3 candidate) and incorporates the
detail into its prose.

**Dependencies / order of work:**

1. Ship the content-string enrichment first (option 2). Confirms phase data flows correctly,
   fixes the immediate report contradiction, no architectural risk.
2. If the LLM proves unable to reason adequately even with the enriched content (e.g. it
   keeps missing subtype hints in titles, or fails to flag adequately powered Phase 2s),
   then add `get_trial_details` and the prompt update.
3. Same architectural pattern likely applies to the literature sub-agent's PubMed tools —
   abstracts on triage, full text on drill-down — but that's a separate piece of work.

Not urgent. Defer until the content-string fix proves insufficient on a real failure case.

## LLM API call reduction (added 2026-04-28)

Disease normalization is already batched where it matters
(`_normalize_disease_groups` in `retrieval.py` collapses ~15 competitor
diseases into one LLM call via `llm_normalize_disease_batch`). Singular
`llm_normalize_disease` calls remain inside `normalize_for_pubmed`, but those
are per-pair and cached by `raw_term`, so repeat-drug runs cost zero. Not a
hot path.

The real LLM-call multiplier is the **sub-agents themselves**. Each literature
ReAct loop and each clinical_trials ReAct loop runs multiple turns with
multiple tool calls. With 5 candidates × 2 sub-agents × N turns each, a
supervisor run is 30+ LLM calls from sub-agent scaffolding alone. Disease
normalization is rounding error against this.

Real targets when API cost matters:

1. **Sub-agent output caching by (drug, disease) pair.** The literature and
   clinical_trials sub-agents do nearly identical work for repeat (drug,
   disease) pairs across runs. Cache the typed `LiteratureOutput` /
   `ClinicalTrialsOutput` artifact keyed by (drug, disease, evidence-cutoff
   date). A repeat dasatinib run becomes ~free for any candidate analyzed
   before. Tradeoff: stale results when new trials/papers land — needs a TTL
   or explicit invalidation.

2. **Anthropic prompt caching for sub-agent system prompts.** Each sub-agent
   has a stable, multi-paragraph system prompt that is sent every turn of
   every ReAct loop. With prompt caching enabled, only the per-call deltas
   count against tokens. Largest token win, no behavior change. Effort: small
   (one SDK call kwarg change in `services/llm.py`).

3. **Reduce sub-agent ReAct turn count.** The literature agent's tool sequence
   is roughly fixed (resolve → queries → fetch → semantic_search → synthesize)
   but the LLM picks the order each time, sometimes loops or backtracks. A
   deterministic pipeline (call tools in fixed order, no LLM-in-the-loop until
   the synthesis step) would cut turns by ~50%. Tradeoff: less adaptive,
   harder to extend with new tools. Worth doing if (1) and (2) aren't enough.

Order of effort: (2) << (1) < (3). Do (2) first whenever caching becomes a
priority — it's a one-line change with no behavioral risk. (1) needs cache
infrastructure and an invalidation story. (3) is a redesign of each sub-agent.

------

APPENDIX


## Scenarios

### Scenario 1 — Drug did not resolve (prerequisite short-circuit)
**Drug:** a made-up string like `"xyzzy-not-a-drug"` or a withdrawn
compound that ChEMBL won't resolve.

**Structural assertions:**
- `output.mechanisms_of_action == []`
- `output.drug_targets == {}`
- `output.shaped_associations == []`

**Summary assertions:**
- Length < 300 chars (single sentence).
- Contains "our tools" or "did not find" (attribute-to-tools rule).
- Must NOT contain: any gene symbol, disease name, or the word
  "hypothesis".
- Must NOT contain banned tokens: `MechanismOfAction`,
  `Association`, `overall_score`, `datatype_scores`.

This proves the prerequisite short-circuit fires and the model
doesn't hallucinate targets from training data.

---

### Scenario 2 — Drug resolves but has no `action_type` on any MoA (direction-unknown short-circuit)
**Drug:** find one in Open Targets where every
`mechanisms_of_action[i].action_type is None`. Candidates to probe:
small-molecule imaging agents or older compounds with incomplete
MoA entries (confirm by inspecting `get_drug` output before
committing to the case).

**Structural assertions:**
- `output.mechanisms_of_action` is non-empty.
- Every `moa.action_type is None`.
- Every `shaped_associations` entry has `shape == "neutral"`.

**Summary assertions:**
- Regex for "direction" AND ("unknown" OR "cannot be determined").
- Must NOT contain "hypothesis" OR "contraindication" (shape words
  are banned when direction is unknown).
- Must NOT claim any repurposing opportunity.

---

### Scenario 3 — Every surfaced association is `confirms_known` (clinical-dominated short-circuit)
**Drug:** a widely-used approved drug whose target–disease
associations on Open Targets are dominated by its own clinical
evidence. Good candidates: `metformin` (type 2 diabetes through
AMPK/PRKAB1 associations) or `atorvastatin` (HMGCR →
hypercholesterolemia). Verify by inspecting
`datatype_scores['clinical'] ≥ 0.6, genetic_association < 0.3` on
the top associations before committing.

**Structural assertions:**
- `shaped_associations` is non-empty.
- `all(a.shape == "confirms_known" for a in output.shaped_associations)`.

**Summary assertions:**
- Contains phrase matching:
  `no (novel|new) .* (hypothesis|repurposing)` OR
  `dominated by .* clinical use`.
- Must NOT use outcome-laden words: `"validated"`, `"promising"`,
  `"encouraging"`, `"would treat"`.
- Must NOT list any disease as a new hypothesis.

This is the analogue of the semaglutide/NASH case — it proves the
model stops rather than pads with "opportunities" when the data
only confirms existing use.

---

### Scenario 4 — Single hypothesis, sufficient score (single-hypothesis branch)
**Drug:** an inhibitor whose target has exactly one
high-`overall_score` non-clinical disease association. Candidate:
`imatinib` (KIT inhibitor → GIST is approved, but KIT also has
mid-score associations to other diseases). Need to pick one where
exactly one pair has `shape=="hypothesis"` and
`overall_score ≥ 0.5` — verify by running
`_compute_shaped_associations` on real data before committing.

**Structural assertions:**
- Exactly one `a` in `shaped_associations` with
  `shape == "hypothesis"` AND `overall_score >= 0.5`.

**Summary assertions:**
- Contains the disease name and the target symbol.
- Contains a numeric score within ±0.02 of the real `overall_score`
  (proves it's citing data, not paraphrasing).
- Plain-English translation of `action_type` (e.g. "inhibits",
  "activates"), NOT the raw token `"INHIBITOR"`.
- Contains exactly one disease name from `shaped_associations`
  where `shape=="hypothesis"` — no extras.

---

### Scenario 5 — Multiple hypotheses (count-scaled branch)
**Drug:** a multi-target kinase inhibitor like `dasatinib` or
`sunitinib` with several targets each having non-clinical
hypothesis-shape associations.

**Structural assertions:**
- `len([a for a in shaped_associations if a.shape == "hypothesis" and a.overall_score >= 0.5]) >= 2`

**Summary assertions:**
- Each hypothesis disease from `shaped_associations` appears in
  the summary (loop assertion).
- Each corresponding `overall_score` appears (within rounding
  tolerance).
- Must NOT use the single-adjective hedge: ban `"moderate evidence"`,
  `"some evidence"`.
- Must NOT collapse multiple hypotheses into one generic sentence —
  assert each disease name shows up individually.

This is the solanezumab-analogue test: prove the model scales with
count instead of flattening.

---

### Scenario 6 — Contraindication shape (directional-opposite branch)
**Drug:** harder to find cleanly. Need an activator where the
disease carries GOF evidence (`somatic_mutation >= 0.4`), e.g. a
kinase activator implicated in a cancer where the target is a
known oncogene. Alternative: construct the case by picking a known
inhibitor + a disease with low somatic/pathway scores and an LOF-name
match (e.g. "…deficiency"). Verify the shape is
`contraindication` in the real output before committing.

**Structural assertions:**
- At least one `a` in `shaped_associations` with
  `shape == "contraindication"`.

**Summary assertions:**
- Uses hedged phrasing like `"directionally inconsistent"` or
  `"opposes"` — regex for `(direction|opposite|inconsistent)`.
- Must NOT say `"contraindicated"` or `"harmful"` (banned — tools
  can't tell you clinical harm).

---

### Scenario 7 — Banned-tokens-everywhere regression (run across all scenarios)
Factor into a parametrized test that runs over every output summary
from the above tests and asserts that none of the banned internal
tokens appear:

```python
BANNED_TOKENS = [
    "MechanismOfAction", "Association", "overall_score", "datatype_scores",
    "genetic_association", "somatic_mutation", "affected_pathway",
    "target_symbols", "shaped_associations", "ShapedAssociation",
    "confirms_known", "GOF", "LOF",
]
# plus outcome-laden words
BANNED_OUTCOME_WORDS = ["validated", "promising", "encouraging",
                        "mechanistically confirmed", "would treat"]
```

---

### Scenario 8 — Grounding regression
**Setup:** Reuse Scenario 4's drug. Before assertions, capture
`surfaced_diseases = {a.disease_name for a in shaped_associations}`
and `surfaced_targets = set(drug_targets.keys())`.

**Summary assertions:**
- Every disease-looking proper noun in the summary ∈
  `surfaced_diseases` (approximate via substring check against the
  set).
- Every gene-symbol-looking token (regex `[A-Z0-9]{2,6}`) that
  appears ∈ `surfaced_targets ∪ {"FDA", "DNA", etc.}`.

This catches training-data leakage directly — the failure mode
RULE 0 was written to prevent.

---

## Test organization notes

- Use `@pytest.fixture` for the agent (same pattern as
  clinical_trials).
- Use `temperature=0, max_tokens=4096` for reproducibility.
- Real API tests are slow and costly — mark them
  `@pytest.mark.integration` and gate on an env var if you want to
  run unit tests fast.
- The structural assertions act as guardrails: if Open Targets
  changes and a case stops exercising its branch, the structural
  assert fails first with a clear message, so you don't waste time
  debugging a summary that was testing the wrong branch.

## What to do before writing the tests

For each scenario, do a dry run of the agent against real data and
inspect `output.shaped_associations` to **confirm the branch fires**.
The clinical_trials test comment for atorvastatin is a good model —
it explains why *that specific drug* exercises the ≥2 branch,
including failed alternatives. Drug pick is the hard part; the
assertions are mechanical once the branch is confirmed.