r"""RAG benchmark: a knowledge base plus hand-labelled questions, run through the production code paths.

By default it uses the shipped sample (data/sample_entries.json + benchmarks/eval-cases.json).
To benchmark your own content, point it at a snapshot of your knowledge base and your own labels
(keep them in the gitignored private/ folder if they contain personal data):

    BENCH_CORPUS=private/benchmarks/corpus.json BENCH_CASES=private/benchmarks/cases.json \\
        python -m benchmarks.evaluate --configs production

Retrieval (offline: local models, in-memory Qdrant):
    python -m benchmarks.evaluate                         # every config, one subprocess each
    python -m benchmarks.evaluate --configs production,+hybrid_bm25
Generation (live Groq from .env, production config, rate-limit friendly):
    python -m benchmarks.evaluate --generation
Results land in BENCH_RESULTS (default benchmarks/results/, gitignored).
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / os.getenv("BENCH_CORPUS", "data/sample_entries.json")
CASES = ROOT / os.getenv("BENCH_CASES", "benchmarks/eval-cases.json")
RESULTS = ROOT / os.getenv("BENCH_RESULTS", "benchmarks/results")
K = 5

PRODUCTION = dict(content_only=False, routing=True)
# Each config is a set of overrides on top of app settings. The "+" chain adds one
# change at a time so each improvement's effect is measured in isolation.
BASELINE = dict(
    content_only=True,
    routing=False,
    embedding_query_prefix="",
    sparse_model="",
    reranker_model="",
    retrieval_score_threshold=0.45,
    retrieval_list_top_k=5,
)
_DENSE_ONLY = dict(BASELINE, content_only=False, embedding_query_prefix=None, retrieval_score_threshold=-1.0)
CONFIGS: dict[str, dict] = {
    "baseline": BASELINE,
    "+query_prefix": dict(BASELINE, embedding_query_prefix=None),
    "+contextual_text": dict(BASELINE, embedding_query_prefix=None, content_only=False),
    "+hybrid_bm25": dict(BASELINE, embedding_query_prefix=None, content_only=False, sparse_model="Qdrant/bm25"),
    "+category_routing": dict(
        BASELINE,
        embedding_query_prefix=None,
        content_only=False,
        sparse_model="Qdrant/bm25",
        routing=True,
        retrieval_list_top_k=10,
    ),
    "+rerank_minilm_l6": dict(PRODUCTION, reranker_model="Xenova/ms-marco-MiniLM-L-6-v2"),
    "+rerank_minilm_l12": dict(PRODUCTION, reranker_model="Xenova/ms-marco-MiniLM-L-12-v2"),
    "+rerank_jina_tiny": dict(PRODUCTION, reranker_model="jinaai/jina-reranker-v1-tiny-en"),
    "+rerank_jina_turbo": dict(PRODUCTION, reranker_model="jinaai/jina-reranker-v1-turbo-en"),
    "production": PRODUCTION,
    "production-no_contextual_text": dict(PRODUCTION, content_only=True),
    "production-no_query_prefix": dict(PRODUCTION, embedding_query_prefix=""),
    # Embedding model comparison: dense-only, same text/prefix handling, no gate.
    "dense:bge-small-en-v1.5": dict(_DENSE_ONLY, embedding_model="BAAI/bge-small-en-v1.5"),
    "dense:bge-base-en-v1.5": dict(_DENSE_ONLY, embedding_model="BAAI/bge-base-en-v1.5"),
    "dense:arctic-embed-xs": dict(_DENSE_ONLY, embedding_model="snowflake/snowflake-arctic-embed-xs"),
    "dense:arctic-embed-s": dict(_DENSE_ONLY, embedding_model="snowflake/snowflake-arctic-embed-s"),
    "dense:all-MiniLM-L6-v2": dict(_DENSE_ONLY, embedding_model="sentence-transformers/all-MiniLM-L6-v2"),
    "dense:jina-v2-small-en": dict(_DENSE_ONLY, embedding_model="jinaai/jina-embeddings-v2-small-en"),
    "dense:nomic-v1.5-Q": dict(_DENSE_ONLY, embedding_model="nomic-ai/nomic-embed-text-v1.5-Q"),
}


def rss_mb() -> float:
    """Return this process's resident memory in MB (Windows working set or Linux VmRSS)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / 2**20
    with open("/proc/self/status") as f:
        return next((int(line.split()[1]) / 1024 for line in f if line.startswith("VmRSS:")), 0.0)


def _load_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the corpus entries (a JSON array or `{"records": [...]}`) and the labelled cases file."""
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    entries = corpus["records"] if isinstance(corpus, dict) else corpus
    return entries, json.loads(CASES.read_text(encoding="utf-8"))


def _apply(overrides: dict[str, Any]) -> tuple[Any, float]:
    """Apply a config to the live settings/services and index the corpus in memory.

    Args:
        overrides: Settings overrides plus the `content_only` / `routing` switches.

    Returns:
        The in-memory Qdrant client and the indexing time in seconds.
    """
    from qdrant_client import QdrantClient

    from app.core.config import settings
    from app.models.knowledge import KnowledgeEntryIn
    from app.services import chunking, query_service, sync_service

    overrides = dict(overrides)
    if overrides.pop("content_only", False):
        chunking.embedding_text = lambda chunk: chunk.text  # reproduces the original pipeline's indexing
    if not overrides.pop("routing", True):
        query_service.route_categories = lambda query, categories: ([], False)
    for key, value in overrides.items():
        setattr(settings, key, value)

    entries, _ = _load_data()
    client = QdrantClient(":memory:")
    started = time.perf_counter()
    chunks = chunking.build_chunks(KnowledgeEntryIn.model_validate(e) for e in entries)
    sync_service.index_chunks(client, chunks)
    return client, time.perf_counter() - started


def _categories(client: Any) -> list[str]:
    """Categories in the index, exactly as the production pipeline sees them."""
    from app.services import embedding_service, qdrant_service

    qdrant_service.clear_category_cache()
    return qdrant_service.list_categories(client, embedding_service.embedding_version())


def _entry_ids(candidates: list[Any]) -> list[str]:
    """Entry ids in rank order, each once (an entry's chunks count as one document)."""
    return list(dict.fromkeys(c.entry_id for c in candidates))


def run_retrieval(name: str) -> dict[str, Any]:
    """Run the retrieval benchmark for one config (offline: local models, in-memory Qdrant).

    Args:
        name: Key in `CONFIGS`.

    Returns:
        Summary metrics, per-type breakdown, latency, memory, threshold calibration
        and per-case rankings.
    """
    from app.core.config import settings
    from app.services import query_service, retrieval_service
    from benchmarks import metrics

    rss_start = rss_mb()
    client, index_seconds = _apply(CONFIGS[name])
    cases = _load_data()[1]["cases"]
    categories = _categories(client)

    # Rank once with the gate disabled, then apply the production select() at the
    # configured threshold and across a sweep, so calibration needs no re-ranking.
    configured = (settings.retrieval_score_threshold, settings.rerank_score_threshold)
    rows, timings = [], {"embed_ms": [], "search_ms": [], "rerank_ms": []}
    for case in cases:
        plan = query_service.plan_query(case["query"], categories=categories)  # offline: no LLM rewrite
        trace: dict = {}
        candidates, reranked = retrieval_service.rank_candidates(plan, client, trace)
        for key in timings:
            timings[key].append(trace.get(key, 0.0))
        rows.append({"case": case, "plan": plan, "candidates": candidates, "reranked": reranked})
    rss_end = rss_mb()

    def evaluate(threshold: float | None = None) -> dict[str, Any]:
        if threshold is not None:
            settings.retrieval_score_threshold = settings.rerank_score_threshold = threshold
        else:
            settings.retrieval_score_threshold, settings.rerank_score_threshold = configured
        per_case = []
        for row in rows:
            selected = retrieval_service.select(row["candidates"], row["reranked"], row["plan"].list_mode)
            per_case.append((row["case"], _entry_ids(selected)))
        settings.retrieval_score_threshold, settings.rerank_score_threshold = configured
        return _summarize(per_case, metrics)

    result = evaluate()
    # Calibration objective: the highest gate that loses no correct hit. Balanced
    # accuracy would weight 6 negatives like 45 positives and refuse real questions.
    ungated = evaluate(-1e9)
    scores = sorted({round(s, 3) for r in rows for s in _scores(r)})
    safe = [
        (t, m)
        for t in scores
        if (m := evaluate(t))[f"hit@{K}"] >= ungated[f"hit@{K}"] and m["positive_false_rejection_rate"] == 0
    ]
    best_t, best = max(safe, key=lambda item: item[0]) if safe else (None, ungated)
    result.update(
        config=name,
        overrides={k: v for k, v in CONFIGS[name].items()},
        index_seconds=round(index_seconds, 2),
        latency_ms={
            k: {"mean": round(metrics.mean(v), 2), "p95": round(metrics.percentile(v, 95), 2)}
            for k, v in timings.items()
        },
        rss_mb={"process": round(rss_end, 1), "models_and_index": round(rss_end - rss_start, 1)},
        calibration={
            "safe_threshold": best_t,
            "negative_abstention": best["negative_abstention_rate"],
            "recall@5_at_safe": best[f"recall@{K}"],
            "recall@5_ungated": ungated[f"recall@{K}"],
            "note": "highest gate losing no Hit@5; few negatives, so indicative only",
        },
        cases=[{"id": c["id"], "ranked": ids} for c, ids in _per_case_ids(rows, retrieval_service)],
    )
    return result


def _scores(row: dict[str, Any]) -> list[float]:
    """Gate-relevant score of every candidate (the gate filters per document)."""
    return [
        c.rerank_score if row["reranked"] else c.dense_score
        for c in row["candidates"]
        if (c.rerank_score if row["reranked"] else c.dense_score) is not None
    ]


def _per_case_ids(rows: list[dict[str, Any]], retrieval_service: Any) -> list[tuple[dict[str, Any], list[str]]]:
    """Return each case with the entry ids that pass the configured gate."""
    return [
        (r["case"], _entry_ids(retrieval_service.select(r["candidates"], r["reranked"], r["plan"].list_mode)))
        for r in rows
    ]


def _summarize(per_case: list[tuple[dict[str, Any], list[str]]], metrics: Any) -> dict[str, Any]:
    """Aggregate per-case rankings into overall, per-type and abstention metrics."""
    positives = [(c, ids) for c, ids in per_case if c["relevant_ids"]]
    negatives = [ids for c, ids in per_case if not c["relevant_ids"]]
    scored = [(c, metrics.retrieval_scores(ids, set(c["relevant_ids"]), K)) for c, ids in positives]
    summary = {key: round(metrics.mean([s[key] for _, s in scored]), 4) for key in scored[0][1]}
    summary["recall@10"] = round(
        metrics.mean([metrics.recall_at_k(ids, set(c["relevant_ids"]), 10) for c, ids in positives]), 4
    )
    by_type: dict[str, list] = {}
    for c, s in scored:
        by_type.setdefault(c["type"], []).append(s)
    summary["by_type"] = {
        t: {k: round(metrics.mean([s[k] for s in rows]), 4) for k in rows[0]} for t, rows in by_type.items()
    }
    summary["negative_abstention_rate"] = round(metrics.mean([1.0 if not ids else 0.0 for ids in negatives]), 4)
    summary["positive_false_rejection_rate"] = round(metrics.mean([1.0 if not ids else 0.0 for _, ids in positives]), 4)
    summary["balanced_accuracy"] = round(
        (summary["negative_abstention_rate"] + 1 - summary["positive_false_rejection_rate"]) / 2, 4
    )
    return summary


def run_generation() -> dict[str, Any]:
    """Run the live generation benchmark on the production config (uses the LLM from .env).

    Returns:
        Summary generation metrics and every case's answer and scores.
    """
    from app.services import qdrant_service, retrieval_service
    from app.services.llm_service import LLMServiceError
    from benchmarks import metrics

    client, _ = _apply(PRODUCTION)
    qdrant_service.get_client = lambda: client
    qdrant_service.clear_category_cache()
    cases_file = _load_data()[1]
    # Optional "generation_cases" list in the cases file limits LLM usage to a subset.
    chosen = set(cases_file.get("generation_cases") or [c["id"] for c in cases_file["cases"]])

    rows = []
    for case in [c for c in cases_file["cases"] if c["id"] in chosen]:
        for _ in range(6):
            try:
                started = time.perf_counter()
                result = retrieval_service.answer_question(case["query"], case.get("history"))
                latency = (time.perf_counter() - started) * 1000
                break
            except LLMServiceError:
                time.sleep(25)
        else:
            raise SystemExit(f"LLM kept failing on {case['id']}")
        relevant = set(case["relevant_ids"])
        returned = list(dict.fromkeys(s["entry_id"] for s in result["sources"]))
        rows.append(
            {
                "id": case["id"],
                "type": case["type"],
                "answer": result["reply"],
                "returned_source_ids": returned,
                "refused": metrics.is_refusal(result["reply"]),
                "cited": bool(metrics.cited_indices(result["reply"])),
                "fact_recall": metrics.fact_recall(result["reply"], case.get("must_include", [])) if relevant else None,
                "source_precision": (len(set(returned) & relevant) / len(returned) if returned else 0.0)
                if relevant
                else None,
                "latency_ms": round(latency, 1),
            }
        )
        time.sleep(9)  # Groq free tier: 8k tokens/minute

    pos = [r for r in rows if r["type"] != "negative"]
    neg = [r for r in rows if r["type"] == "negative"]
    summary = {
        "cases": len(rows),
        "negative_refusal_rate": round(metrics.mean([1.0 if r["refused"] else 0.0 for r in neg]), 4),
        "positive_false_refusal_rate": round(metrics.mean([1.0 if r["refused"] else 0.0 for r in pos]), 4),
        "fact_recall": round(metrics.mean([r["fact_recall"] for r in pos]), 4),
        "source_precision": round(metrics.mean([r["source_precision"] for r in pos]), 4),
        "answers_with_citations": round(metrics.mean([1.0 if r["cited"] else 0.0 for r in pos]), 4),
        "end_to_end_latency_ms_mean": round(metrics.mean([r["latency_ms"] for r in rows]), 1),
    }
    return {"config": "production", "generation": summary, "cases": rows}


def _quiet() -> None:
    """Silence warnings and app logging so stdout carries only results."""
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)


def main() -> None:
    """Command-line entry point (see the module docstring for usage)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", help="comma-separated config names (default: all)")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--generation", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.one:
        _quiet()
        print(json.dumps(run_retrieval(args.one)))
        return
    if args.generation:
        _quiet()
        result = run_generation()
        (RESULTS / "generation-production.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result["generation"], indent=2))
        return

    names = args.configs.split(",") if args.configs else list(CONFIGS)
    summary = {}
    for name in names:
        proc = subprocess.run(
            [sys.executable, "-m", "benchmarks.evaluate", "--one", name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proc.returncode:
            print(f"{name}: FAILED\n{proc.stderr[-2000:]}")
            continue
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        (RESULTS / f"retrieval-{name.replace(':', '_').replace('+', 'plus_')}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary[name] = result
        print(
            f"{name:28s} hit@5={result['hit@5']:.3f} recall@5={result['recall@5']:.3f} "
            f"recall@10={result['recall@10']:.3f} "
            f"P@5={result['precision@5']:.3f} mrr={result['mrr']:.3f} ndcg@5={result['ndcg@5']:.3f} "
            f"abstain={result['negative_abstention_rate']:.2f} rss={result['rss_mb']['models_and_index']}MB "
            f"embed={result['latency_ms']['embed_ms']['mean']}ms rerank={result['latency_ms']['rerank_ms']['mean']}ms",
            flush=True,
        )
    existing = (
        json.loads((RESULTS / "retrieval-summary.json").read_text(encoding="utf-8"))
        if (RESULTS / "retrieval-summary.json").exists()
        else {}
    )
    existing.update({n: {k: v for k, v in r.items() if k != "cases"} for n, r in summary.items()})
    (RESULTS / "retrieval-summary.json").write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
