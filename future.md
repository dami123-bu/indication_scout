# Future Improvements

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
