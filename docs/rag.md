# IndicationScout RAG Pipeline

## Why RAG Is Needed

During Path 2 validation testing, we confirmed that PubMed returns ~100 papers per drug-disease query, but many are irrelevant. For example, searching `bupropion AND obesity` returns depression papers that incidentally mention obesity — burying the papers about bupropion as an actual *treatment* for obesity.

The RAG pipeline solves this by embedding and reranking retrieved abstracts so the Literature Agent receives the most relevant papers, not just keyword matches.

**Confirmed empirically with:**
- Bupropion → obesity: papers about immunometabolic depression dominated results; the actual Contrave (naltrexone/bupropion) papers were buried
- Sildenafil → diabetic nephropathy: needed reranking to separate PDE5-specific evidence from noisy non-selective PDE inhibitor literature
- Baricitinib → myelofibrosis: needed reranking to distinguish JAK1/2 inhibition therapeutic context from general oncology mentions

---

## Architecture Overview

```
PubMed E-utilities
       │
       ▼
┌──────────────────┐
│  Fetch abstracts  │  (existing PubMed client)
│  for drug+disease │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Store in         │  PostgreSQL + pgvector
│  pgvector cache   │  (deduplicate by PMID)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Embed abstracts  │  Voyage AI or OpenAI embeddings
│  + embed query    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vector similarity│  Cosine similarity reranking
│  reranking        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Top-k papers to  │  Feed to Literature Agent
│  Literature Agent │  for LLM synthesis (Claude)
└──────────────────┘
```

---

## Components

### 1. PostgreSQL + pgvector (Abstract Cache + Vector Store)

Serves two purposes:
- **Caching**: Avoid re-fetching the same PubMed abstracts across runs. Deduplicate by PMID.
- **Vector search**: Store embeddings alongside abstracts for semantic retrieval.

**Proposed schema:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE pubmed_abstracts (
    pmid          TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    abstract      TEXT,
    authors       TEXT[],
    journal       TEXT,
    pub_date      TEXT,
    mesh_terms    TEXT[],
    keywords      TEXT[],
    embedding     vector(1024),   -- dimension depends on embedding model
    fetched_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON pubmed_abstracts USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Track which drug-disease queries have been run
CREATE TABLE search_queries (
    id            SERIAL PRIMARY KEY,
    drug_name     TEXT NOT NULL,
    disease_name  TEXT NOT NULL,
    pubmed_query  TEXT NOT NULL,
    pmids         TEXT[],
    searched_at   TIMESTAMP DEFAULT NOW(),
    UNIQUE(drug_name, disease_name)
);
```

**Why pgvector over a dedicated vector DB (Pinecone, Weaviate, etc.):**
- Single infrastructure dependency (Postgres already needed)
- Abstracts are small enough (~10k-50k total across all drugs) that pgvector performance is fine
- Simpler Docker setup: one `postgres:16` container with `pgvector` extension

### 2. Embedding API

**Options discussed:**
- **Voyage AI** (`voyage-3`) — optimized for scientific/biomedical retrieval
- **OpenAI** (`text-embedding-3-small` or `text-embedding-3-large`)

The embedding model encodes both the PubMed abstracts (stored in pgvector) and the retrieval query at search time.

**Query formulation for embedding:**
The query should encode the therapeutic intent, not just keywords. For example:
- Keyword query: `"bupropion AND obesity"`
- Embedding query: `"Evidence for bupropion as a treatment for obesity, including clinical trials, efficacy data, and mechanism of action"`

This semantic framing is what separates papers about bupropion *treating* obesity from papers that mention both words in a depression context.

### 3. Retrieval + Reranking Flow

```python
async def retrieve_and_rerank(
    drug_name: str,
    disease_name: str,
    pubmed_query: str,
    top_k: int = 10
) -> list[PubMedArticle]:
    """
    1. Check cache for existing results
    2. If miss, fetch from PubMed (up to 100 abstracts)
    3. Store abstracts + embeddings in pgvector
    4. Embed the therapeutic query
    5. Rerank by cosine similarity
    6. Return top-k
    """
    
    # Step 1: Check if we've already fetched for this drug-disease pair
    cached = await check_search_cache(drug_name, disease_name)
    if not cached:
        # Step 2: Fetch from PubMed
        articles = await pubmed_client.search(pubmed_query, max_results=100)
        
        # Step 3: Embed and store
        for article in articles:
            if not await abstract_exists(article.pmid):
                embedding = await embed(article.title + " " + article.abstract)
                await store_abstract(article, embedding)
        
        await record_search(drug_name, disease_name, pubmed_query, 
                          [a.pmid for a in articles])
    
    # Step 4: Build therapeutic query embedding
    therapeutic_query = (
        f"Evidence for {drug_name} as a treatment for {disease_name}, "
        f"including clinical trials, efficacy, mechanism of action, "
        f"and therapeutic outcomes"
    )
    query_embedding = await embed(therapeutic_query)
    
    # Step 5: Vector similarity search over cached PMIDs
    pmids = await get_cached_pmids(drug_name, disease_name)
    results = await vector_search(query_embedding, pmids, top_k=top_k)
    
    return results
```

### 4. Disease Name Normalization (Pre-RAG Step)

Open Targets EFO disease names don't always match PubMed indexing. Confirmed issues:
- `"narcolepsy-cataplexy syndrome"` → PubMed needs `"narcolepsy"`
- `"eczematoid dermatitis"` → PubMed needs `"atopic dermatitis"`
- `"type 2 diabetes nephropathy"` → PubMed needs `"diabetic nephropathy"`

**Solution:** One LLM call (Claude Haiku) per candidate disease before PubMed search:

```python
async def normalize_disease_for_pubmed(efo_disease_name: str) -> str:
    prompt = f"""Convert this disease name to the best PubMed search query.
    Return ONLY the search term, nothing else.
    Disease: {efo_disease_name}"""
    
    return await haiku_call(prompt)
```

Run for all 10 candidates upfront — 10 cheap Haiku calls. No retry logic, no synonym explosion. The LLM already knows the PubMed-friendly terms.

---

## Where RAG Fits in the Full Pipeline

```
┌─────────────┐     ┌─────────────┐
│   Path 1     │     │   Path 2     │
│  (target-    │     │  (drug class │
│   disease    │     │   analogy)   │
│   assoc.)    │     │              │
└──────┬───────┘     └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
          ┌───────────────┐
          │   Merge &      │  Deduplicate by EFO ID,
          │   Rank Top 10  │  flag dual-path candidates
          └───────┬────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │     For each candidate:      │
    │                              │
    │  1. Normalize disease name   │  ← LLM (Haiku)
    │  2. PubMed search            │  ← Existing client
    │  3. RAG rerank               │  ← pgvector + embeddings
    │  4. Literature Agent synth.  │  ← LLM (Claude)
    │  5. Clinical Trial search    │  ← Existing client
    │  6. Trial Agent synthesis    │  ← LLM (Claude)
    │                              │
    └─────────────┬────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  Evidence      │  Combine Path scores +
          │  Scoring &     │  literature + trial evidence
          │  Report Gen    │
          └────────────────┘
```

The RAG pipeline sits between raw PubMed retrieval and the Literature Agent. It does not change the discovery logic (Path 1 / Path 2) — it improves the quality of evidence presented to the LLM for synthesis.

---

## Infrastructure Setup (Docker)

```yaml
# docker-compose.yml addition
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: indication_scout
      POSTGRES_USER: scout
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## Sprint Mapping

| Task | Sprint | Status |
|------|--------|--------|
| Docker + pgvector setup | Sprint 1 | Not started |
| Abstract caching schema | Sprint 1 | Not started |
| Embedding integration (Voyage/OpenAI) | Sprint 1 | Not started |
| Reranking function | Sprint 1-2 | Not started |
| Disease name normalization (Haiku) | Sprint 2 | Not started |
| Literature Agent integration | Sprint 2 | Not started |
| Full pipeline wiring via LangGraph | Sprint 2-3 | Not started |

---

## Key Design Decisions

1. **pgvector over dedicated vector DB** — simplicity; dataset is small enough (~10k-50k abstracts)
2. **Therapeutic query framing** — embed intent ("evidence for X as treatment for Y") not just keywords
3. **Cache-first retrieval** — avoid redundant PubMed API calls and re-embedding
4. **LLM disease name normalization** — cheap Haiku calls instead of building a synonym dictionary or ontology traversal
5. **Rerank within PubMed result set** — not open-ended vector search across all stored abstracts; scope reranking to the PMIDs returned by the drug-disease PubMed query

---

## Validated Test Cases

These drug-disease pairs confirmed that PubMed has signal but reranking is needed:

| Drug | Candidate Disease | Issue Without RAG |
|------|-------------------|-------------------|
| Bupropion | Obesity | Depression papers mentioning obesity dominate; Contrave papers buried |
| Bupropion | Narcolepsy | EFO name mismatch (`narcolepsy-cataplexy syndrome`) returned zero results |
| Sildenafil | BPH | Confirmed signal (PMID 40678732, 38448685) but mixed with erectile dysfunction papers |
| Sildenafil | Diabetic nephropathy | Needed reranking to separate PDE5-specific from non-selective PDE literature |
| Baricitinib | Eczematoid dermatitis | EFO name doesn't match PubMed indexing (`atopic dermatitis`) |