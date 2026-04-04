# LiteratureAgent

## Purpose

`LiteratureAgent` finds and synthesizes PubMed evidence for a drug-disease pair. It returns
an `EvidenceSummary` describing the strength, volume, and nature of published evidence.

It is a LangChain ReAct agent (same pattern as `ClinicalTrialsAgent`).

**File:** [src/indication_scout/agents/literature.py](../src/indication_scout/agents/literature.py)

---

## Inputs and Outputs

**Input (`input_data`):**

| Key | Type | Description |
|-----|------|-------------|
| `drug_name` | `str` | Common drug name (e.g. `"metformin"`) |
| `disease_name` | `str` | Candidate indication (e.g. `"colorectal cancer"`) |
| `drug_profile` | `DrugProfile` | Pre-built drug profile (synonyms, targets, MOA, ATC codes) |

**Output:**

| Key | Type | Description |
|-----|------|-------------|
| `evidence_summary` | `EvidenceSummary` | Synthesized evidence (strength, study count, key findings, PMIDs) |
| `pmids_retrieved` | `list[str]` | All PMIDs fetched during the run |

---

## Tools

Three tools wrapping `RetrievalService`, defined in `agents/literature_tools.py`.
`drug_profile` is captured via closure at build time.

| Tool | What it does |
|---|---|
| `expand_search_terms(drug_name, disease_name)` | Generates 5–10 PubMed queries using synonyms, targets, MOA, ATC codes |
| `fetch_and_cache(queries)` | Runs PubMed searches, embeds with BioLORD-2023, stores in pgvector. Returns `pmid_count` and `pmids` |
| `semantic_search(pmids, top_k=5)` | Retrieves top-k most similar abstracts from pgvector. Returns abstracts with similarity scores |
| `synthesize(abstracts)` | Passes abstracts to LLM, returns `EvidenceSummary` |

## Fixed call sequence

No branching or retry logic. The agent always calls tools in this order:

```
expand_search_terms → fetch_and_cache → semantic_search → synthesize
```

If no evidence is found, `synthesize` is still called and returns `strength: "none"`.

---

## Dependencies

| Component | Role |
|-----------|------|
| `RetrievalService` | Executes all four tool operations |
| `DrugProfile` | Provides synonyms, targets, MOA, ATC codes for query expansion |
| `EvidenceSummary` | Output model (`models/model_evidence_summary.py`) |

---

## What Needs to Be Built

- [ ] `agents/literature_tools.py` — `@tool` wrappers around `RetrievalService`
- [ ] `LiteratureAgent.run()` — `create_agent` + `ainvoke` + `_parse_result`
- [ ] System prompt
- [ ] Unit tests

See [mock_tools.md](../mock_tools.md) for the implementation mockup.
