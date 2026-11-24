# RAG Architecture Assessment

> **Context:** engineering notes from this project's first deployment, a single-person portfolio assistant (the owner is called "Alex" here). That corpus and its labels are private and not shipped; the shipped benchmark (`python -m benchmarks.evaluate`) runs on `data/sample_entries.json`. Re-run on that private corpus, the current generic pipeline reproduces the final production numbers in [rag-improvement-roadmap.md](rag-improvement-roadmap.md) exactly.

## The system being optimised

A single-tenant chatbot for one public portfolio. That shapes every decision below.

| Property | Value |
|---|---|
| Knowledge source | One JSON file in Supabase Storage, 35 curated records, 7 categories |
| Record size | 61–222 tokens incl. title and category (median 104) — embedding limit is 512 |
| Users | Anonymous visitors; all content is public by design |
| Hosting | Render free tier: 512 MB RAM, 0.1 CPU, sleeps after 15 min |
| LLM budget | Groq free tier: 8k tokens/min, ~200k tokens/day |

## Original request flow (before)

```
POST /api/v1/chat/ask {question}
 -> embed_query(question)              raw query; fastembed 0.5.1 adds NO instruction prefix
 -> Qdrant dense cosine search         top-5, score >= 0.45, unnamed single vector
 -> context = "[category] title: content" x5, concatenated
 -> one *user* message: rules + context + question   (Groq default temperature)
 -> {answer, sources = all 5 retrieved records}
```

Ingestion: Supabase JSON -> `build_documents` -> embed **`content` only** -> upsert by
`uuid5(record id)`; skip records whose hash is unchanged. Sync ran on the event
loop, blocking every chat request while it embedded.

## Component inventory (before)

| Component | State | Assessment |
|---|---|---|
| Extraction / formats | JSON records only | Correct for this source |
| Chunking | None — one record = one vector | Correct: records are atomic and fit the model |
| Embedding | bge-small-en-v1.5, 384-d, `EMBEDDING_DIM = 384` hardcoded | Changing the model would have broken indexing |
| Embedded text | `content` only | Titles unused — FAQ titles are literally the questions |
| Query embedding | No BGE retrieval instruction | Recommended for short query -> passage search |
| Retrieval | Dense cosine only | Misses rare identifiers (Tidewatch) |
| Metadata filtering | `category` filter existed, never used | — |
| Reranking | None | — |
| Evidence gate | Cosine >= 0.45 | Rejected 0% of off-topic questions (see baseline) |
| Context | Concatenated top-5 | No numbering, budget, or attribution |
| Prompt | Rules and data in one user message | Weaker instruction hierarchy; default temperature |
| Citations | None; all 5 sources returned | 70% of returned sources irrelevant |
| Conversation | Stateless, single question | Follow-ups unanswerable |
| LLM resilience | No retries | One Groq 429 = failed request |
| Errors | Qdrant outages surfaced as 500 | Should be 503 like other dependencies |
| Observability | Plain logs, no request id, no stage timings | — |
| Security | Gemini key sent as `?key=` | httpx logs request URLs: key would be written to logs |

## Root causes of the measured weaknesses

1. **List questions (recall 0.36):** with content-only embeddings and one global
   top-5, "projects" questions matched bios and skill summaries, whose text
   says "built projects", rather than the project records themselves.
2. **Exact identifiers:** dense embeddings blur rare tokens like "Tidewatch";
   nothing lexical backed them up.
3. **No out-of-scope protection:** every record names the owner, so any
   personal question scores 0.59–0.70 cosine. A cosine floor cannot separate
   "about Alex and answerable" from "about Alex but not in the knowledge base".
   Only a model that reads query and passage together (a cross-encoder) can.
4. **Follow-ups:** no history, so references had nothing to resolve against.
5. **Source noise:** sources were "everything retrieved", not "what the answer used".

## Parts of the brief that do not apply here — and when they would

| Brief item | Decision | Reason | Revisit when |
|---|---|---|---|
| PDF/DOCX/XLSX extraction | Not built | Source is curated JSON; no files to parse | Documents are added as files |
| Semantic / structure-aware chunking, overlap | Not built | Longest record is 222 tokens, under half the 512 limit; splitting atomic facts only loses context | Records exceed ~400 tokens |
| Parent-child retrieval | Not built | The record *is* the parent; nothing larger to expand to | Same as chunking |
| Tenant / document ACL | Not built | One tenant; every record is public by design | Private content is added |
| Document version history | Fingerprints only | Supabase is the source of truth; Qdrant is rebuildable | — |
| Embedding-provider abstraction | Config only | FastEmbed (local, free) is the only provider allowed by the project spec; the model is already a setting | A hosted embedding API is approved |
| LLM multi-query / multi-hop | Not built | Hybrid + routing + reranking reach the multi-category records (see results); each extra LLM call costs ~1.3k of an 8k tokens/min budget | Multi-category recall regresses |
| 10-label query classifier | Only behaviour-changing labels | Follow-up -> rewrite, list -> wider k and category routing, out-of-scope -> evidence gate. Other labels would change nothing | A label starts driving a different strategy |
| RAGAS / DeepEval | Deterministic metrics instead | Heavy dependencies plus an LLM judge call per metric per case | Budget allows an LLM judge |
| Embedding/query cache beyond process | In-process LRU | Single Render instance | Horizontal scaling |
