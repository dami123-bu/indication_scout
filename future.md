# Future Improvements

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

### 2. Use MeSH terms for indications
ClinicalTrials.gov indexes conditions against MeSH. Resolving the indication to its
MeSH preferred term before querying gives much better recall. The NLM has a free
MeSH lookup API.

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
