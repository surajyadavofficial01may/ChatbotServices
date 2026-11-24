# RAG Improvement Roadmap — Decisions, Results, Next Steps

> **Context:** engineering notes from this project's first deployment, a single-person portfolio assistant (the owner is called "Alex" here). That corpus and its labels are private and not shipped; the shipped benchmark (`python -m benchmarks.evaluate`) runs on `data/sample_entries.json`. Re-run on that private corpus, the current generic pipeline reproduces the final production numbers in the results below exactly.

Companion to [rag-architecture-assessment.md](rag-architecture-assessment.md) (the
"before" picture) and [rag-benchmark-baseline.md](rag-benchmark-baseline.md) (the
baseline numbers). Everything here was measured with `benchmarks/evaluate.py` on
the same 51-case dataset and 35-record corpus (both private; see the note above).

## Before / after

_Production numbers re-measured after redacting personal contact details from the benchmark corpus (differences ≤ 0.004)._

### Retrieval (51 cases, offline, real models)

| Metric | Baseline | Production | Change |
|---|---|---|---|
| Hit@5 | 0.889 | **0.978** | +0.089 |
| Recall@5 | 0.781 | **0.884** | +0.103 |
| Recall@10 | 0.781 | **0.896** | +0.115 |
| Precision@5 | 0.302 | **0.382** | +0.080 |
| MRR | 0.819 | **0.941** | +0.122 |
| nDCG@5 | 0.761 | **0.883** | +0.122 |
| Answerable questions rejected by the gate | 0.00 | 0.00 | — |

| Type | Hit@5 before → after | Recall@5 before → after | MRR before → after |
|---|---|---|---|
| exact | 0.94 → **1.00** | 0.85 → **1.00** | 0.94 → **1.00** |
| semantic | 0.93 → **1.00** | 0.90 → 0.88 | 0.86 → **1.00** |
| list | 0.80 → **1.00** | 0.36 → **0.65** | 0.57 → **0.90** |
| multi | 0.67 → **1.00** | 0.56 → **0.89** | 0.67 → **0.78** |
| typo | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |
| followup (offline, no rewrite) | 0.67 → 0.67 | 0.50 → 0.50 | 0.40 → 0.50 |

Follow-ups are resolved by an LLM rewrite, which the offline run skips, so their
real effect shows in the generation table: all three now answer correctly.

### Generation (27 cases, live `openai/gpt-oss-120b` on Groq)

| Metric | Baseline | Production |
|---|---|---|
| Off-topic questions refused | 1.00 | 1.00 |
| Answerable questions refused | 0.095 | **0.00** |
| Fact recall (hand-picked facts present) | 0.905 | **1.00** |
| Source precision (returned sources that are relevant) | 0.295 | **0.972** |
| Answers citing their sources | 0.00 | **1.00** |
| Mean end-to-end latency (local models + Groq) | 1450 ms | 1854 ms |

Both rows were scored with the same functions. Refusals count either the "not
available" sentence or an explicit decline. The prompt-injection case is declined
by both systems, and neither leaks the system prompt.

**Latency cost: +404 ms mean.** Drivers: the extra LLM rewrite round-trip on
follow-ups (3 of 27 cases), longer context on list questions (up to 10 sources),
a longer system prompt, and normal Groq variance. Retrieval itself costs
5.7 ms embedding + 1.8 ms search (in-memory) and stays negligible.

Two scoring artifacts were found and fixed during evaluation, then applied to
both systems. The model writes "81 % accuracy" with a narrow no-break space, so
facts are matched whitespace-insensitively. It also declines injections with
"I can't comply" rather than the stock sentence.

## What changed and why (each step measured in isolation)

| Step | Hit@5 | Recall@5 | MRR | nDCG@5 | Verdict |
|---|---|---|---|---|---|
| baseline | 0.889 | 0.781 | 0.819 | 0.761 | — |
| + BGE query instruction | 0.889 | 0.798 | 0.808 | 0.767 | keep: small gain, model-card recommendation (fastembed 0.5.1 never applied it) |
| + category/title in embedded text | 0.889 | 0.771 | 0.822 | 0.751 | neutral alone, essential with BM25 (see ablation) |
| + hybrid BM25 + RRF | 0.956 | 0.863 | 0.918 | 0.857 | **keep: largest single gain** |
| + list-question routing | **0.978** | **0.884** | **0.941** | **0.883** | **keep** = production |
| production without category/title text | 0.978 | 0.837 | 0.877 | 0.815 | title text is worth +0.064 MRR |
| production without query instruction | 0.978 | 0.871 | 0.930 | 0.873 | instruction is worth +0.011 MRR |

**Routing lesson.** The first version routed *every* keyword match. That lifted
list recall but collapsed multi-part questions (Hit@5 1.00 → 0.33):
"What technologies did he use for Tidewatch?" was sent to Technical Skills and
the Tidewatch records were crowded out. Routing now applies only to "list all X"
questions, which fixed both.

### Embedding model comparison (dense-only, same text and prefix handling)

| Model | Dim | Hit@5 | Recall@5 | MRR | nDCG@5 | Added RSS |
|---|---|---|---|---|---|---|
| **BAAI/bge-small-en-v1.5** (kept) | 384 | 0.889 | 0.771 | **0.822** | 0.751 | 330 MB |
| snowflake-arctic-embed-s | 384 | **0.933** | **0.802** | 0.804 | **0.755** | 383 MB |
| snowflake-arctic-embed-xs | 384 | 0.867 | 0.743 | 0.767 | 0.712 | 346 MB |
| all-MiniLM-L6-v2 | 384 | 0.911 | 0.756 | 0.736 | 0.677 | 347 MB |
| jina-embeddings-v2-small-en | 512 | 0.889 | 0.746 | 0.780 | 0.717 | 473 MB |
| nomic-embed-text-v1.5-Q | 768 | 0.867 | 0.761 | 0.797 | 0.740 | 1182 MB |
| BAAI/bge-base-en-v1.5 | 768 | 0.822 | 0.727 | 0.756 | 0.714 | 799 MB |

`bge-small` was kept. arctic-embed-s trades a little MRR for a little recall, a
difference of one or two queries on 45 positives, so it isn't worth +50 MB.
Bigger models were *worse* here and don't fit 512 MB. Hybrid BM25 added far more
than any model swap.

### Rerankers — implemented, evaluated, off by default

| Reranker | Hit@5 | MRR | Added RSS | Rerank latency (desktop CPU) |
|---|---|---|---|---|
| none (production) | **0.978** | **0.941** | — | — |
| ms-marco-MiniLM-L-6-v2 | 0.889 | 0.826 | +372 MB | ~1.0–1.5 s |
| ms-marco-MiniLM-L-12-v2 | 0.933 | 0.832 | +420 MB | ~2.8 s |
| jina-reranker-v1-tiny-en | 0.933 | 0.850 | +437 MB | ~1.3 s |
| jina-reranker-v1-turbo-en | 0.956 | 0.832 | +455 MB | ~1.6 s |

These general web-search cross-encoders *lowered* ranking quality on this corpus,
most of all for list and follow-up questions, where they demote everything but
the single closest record. They also can't fit Render free (512 MB, 0.1 CPU).
Reranking stays available (`RERANKER_MODEL`), with a fallback to fusion order if
the model fails; `-11` is its calibrated gate for MiniLM-L-6.

### Evidence gate

Every record names the owner, so off-topic personal questions score 0.47–0.66
cosine while the weakest correct answer (Tidewatch, found via BM25) scores 0.477.
No cosine gate can separate them without refusing real questions. The 0.45 floor
is kept because it loses zero answers and trims noise, but the actual
out-of-scope guard is the answer prompt's refusal rule (measured: 100%).

## Implemented

- **Embedding layer** (`embedding_service.py`): per-model query/document
  prefixes; dimension read from the model registry and validated, replacing the
  hardcoded 384; thread-safe single model load; LRU query cache; an
  *embedding version* stamped on every vector and used to filter queries, so
  vectors from different models never mix.
- **Ingestion** (`chunking.embedding_text`, used by `sync_service.index_chunks`): embeds "category: title\ncontent";
  fingerprint = embedding version + all indexed fields, so only changed records are
  re-embedded and a model change re-embeds everything; schema check recreates
  the collection on dimension/vector-layout changes; runs off the event loop.
- **Hybrid retrieval** (`qdrant_service.query_candidates`): dense + BM25 + optional
  category retrievers in one Qdrant round-trip, fused with RRF.
- **Query layer** (`query_service.py`): follow-up detection, then LLM rewrite to a
  standalone query (falls back to the original on failure); list-question
  routing with a wider top-k.
- **Reranking** (`rerank_service.py`): optional cross-encoder with graceful fallback.
- **Context and generation**: numbered, attributed, token-budgeted context; system
  and user messages separated; temperature 0.2; grounding, citation, partial-answer,
  conflict and injection rules; citations renumbered (including gpt-oss `【1】` style)
  so `[n]` maps to `sources[n-1]`; only cited sources are returned.
- **Conversation**: optional client-side `history`, used only to rewrite the
  query, never as evidence. Nothing is stored server-side, so there is no
  cross-visitor leakage.
- **Resilience**: LLM retries with exponential backoff honouring `Retry-After`;
  Qdrant outages return 503.
- **Observability**: `X-Request-ID` on every request and in every log line; one
  `rag_request` log line with per-stage timings, retriever counts, top score and
  refusal flag (never the question text); `RAG_DEBUG` exposes the full trace.
- **Security**: Gemini key moved from URL to header (httpx logs URLs);
  sanitised request IDs; third-party request logging silenced.
- **Evaluation**: 51-case benchmark, deterministic metrics, ablation and model
  matrix, threshold calibration, live generation scoring, and a regression-gate
  test suite (`RUN_MODEL_TESTS=1`).

## Next steps — with the trigger that makes each worthwhile

| Item | Trigger |
|---|---|
| List recall beyond 0.65 ("list all projects" still misses 2 of 8) | Raise `RETRIEVAL_LIST_TOP_K` to 12 if the content grows, or answer list questions from a category scan instead of similarity search |
| Domain-tuned reranker (e.g. bge-reranker-base) | Moving to a host with ≥2 GB RAM; re-run `+rerank_*` configs first |
| LLM-judge faithfulness (RAGAS-style) | A paid LLM tier — the 8k tokens/min budget can't absorb it |
| Chunking / parent-child | Records growing past ~400 tokens, or long-form documents added |
| Document-level access control | Any non-public content entering the knowledge base |
| Cross-process caches / sync lock | Running more than one Render instance |

## Deploying this change

1. Deploy the new code. No new environment variables are required: every new
   setting has a benchmarked default (see `.env.example`).
2. Call `POST /api/v1/admin/sync` once. The collection moves from one unnamed
   vector to named `dense` + `bm25` vectors. Sync detects the old schema,
   recreates the collection and re-indexes from Supabase. This was verified
   against real Qdrant Cloud on a temporary collection: 35 records indexed in
   7.8 s, re-sync a no-op. Until sync runs, `/chat/ask` returns 503 (Qdrant
   rejects queries against the old layout) — it never mixes vector formats.
   After a later model change with the same layout, queries match no vectors of
   the new version, so answers are "not available" until the next sync.
3. Optional frontend upgrade: send `history` for follow-ups and render `[n]` as
   links to `sources[n-1]`. Existing clients keep working unchanged.
4. Check RSS on the Render instance after the first questions; locally the
   stack adds ~330 MB.
