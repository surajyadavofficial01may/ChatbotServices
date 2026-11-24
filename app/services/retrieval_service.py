"""The RAG pipeline: plan, retrieve (hybrid + fusion + optional rerank), gate, generate, cite."""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from app.core.config import settings
from app.services import embedding_service, qdrant_service, rerank_service
from app.services.llm_service import LLMServiceError, generate_answer
from app.services.query_service import QueryPlan, plan_query

logger = logging.getLogger(__name__)

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# gpt-oss models often fall back to their native "【1】" / "【1†L1-L3】" citation style.
NATIVE_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")
SNIPPET_CHARS = 500


@dataclass
class Candidate:
    """A retrieved chunk and the scores that ranked it.

    Attributes:
        entry_id: Source entry id.
        chunk_index: Position of the chunk within its entry.
        category: Entry category.
        title: Entry title.
        content: Chunk text.
        url: Optional source link ("" for none).
        rrf: Reciprocal Rank Fusion score across retrievers.
        dense_score: Cosine similarity from a dense retriever, if one found the chunk.
        rerank_score: Cross-encoder score, when reranking is enabled.
        ranks: Rank of the chunk in each retriever that returned it.
    """

    entry_id: str
    chunk_index: int
    category: str
    title: str
    content: str
    url: str
    rrf: float = 0.0
    dense_score: float | None = None
    rerank_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Unique id of the chunk."""
        return f"{self.entry_id}#{self.chunk_index}"


def answer_question(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Answer a question from the knowledge base.

    Args:
        question: The user's question.
        history: Optional prior turns, used only to resolve follow-up references.

    Returns:
        `{"reply", "sources"}` — sources are the chunks the reply cites, in citation
        order — plus `"debug"` when `RAG_DEBUG` is on.

    Raises:
        ExternalServiceError: Subclasses (Qdrant, embedding, LLM) when a dependency fails.
    """
    started = time.perf_counter()
    trace: dict[str, Any] = {}
    client = qdrant_service.get_client()

    t = time.perf_counter()
    categories = qdrant_service.list_categories(client, embedding_service.embedding_version())
    plan = plan_query(question, history, categories)
    trace["plan_ms"] = _ms(t)

    candidates = retrieve(plan, client, trace=trace)
    if not candidates:
        reply, sources = settings.no_answer_message, []
    else:
        context, used, trace["context_tokens"] = build_context(candidates, settings.max_context_tokens)
        t = time.perf_counter()
        raw = generate_answer(question, context, plan.retrieval_query if plan.rewritten else None)
        trace["llm_ms"] = _ms(t)
        if not raw:
            raise LLMServiceError("LLM returned an empty answer.")
        reply, cited = apply_citations(raw, used)
        # Sources = evidence the reply actually cited. An uncited reply (refusal,
        # declined injection) returns none rather than every retrieved chunk.
        sources = [] if _is_refusal(reply) else cited

    trace["total_ms"] = _ms(started)
    trace["refused"] = not sources
    _log_trace(plan, trace)

    result: dict[str, Any] = {"reply": reply, "sources": [_source(c) for c in sources]}
    if settings.rag_debug:
        result["debug"] = {
            "retrieval_query": plan.retrieval_query,
            "categories": plan.categories,
            "list_mode": plan.list_mode,
            **trace,
        }
    return result


def retrieve(plan: QueryPlan, client: QdrantClient, trace: dict[str, Any] | None = None) -> list[Candidate]:
    """Retrieve, rank and gate evidence for a query plan.

    Args:
        plan: The query plan.
        client: Qdrant client.
        trace: Optional dict that receives timings and diagnostics.

    Returns:
        The chunks that passed the evidence gate, best first, capped at top-k.
    """
    trace = trace if trace is not None else {}
    candidates, reranked = rank_candidates(plan, client, trace)
    selected = select(candidates, reranked, plan.list_mode)
    trace["selected"] = len(selected)
    trace["top_score"] = round(_score(selected[0]), 4) if selected else None
    if settings.rag_debug:
        trace["candidates"] = [
            {
                "id": c.key,
                "title": c.title,
                "ranks": c.ranks,
                "rrf": round(c.rrf, 5),
                "dense": _round(c.dense_score),
                "rerank": _round(c.rerank_score),
                "selected": c in selected,
            }
            for c in candidates
        ]
    return selected


def select(candidates: list[Candidate], reranked: bool, list_mode: bool) -> list[Candidate]:
    """Apply the evidence gate and top-k.

    With a reranker the cross-encoder score decides relevance; without one, the dense
    cosine does (chunks found only by BM25 have no cosine and are dropped).

    Args:
        candidates: Candidates in relevance order.
        reranked: Whether the order and scores come from the reranker.
        list_mode: Use the wider top-k for "list all X" questions.

    Returns:
        The kept candidates, best first.
    """
    if reranked:
        kept = [
            c for c in candidates if c.rerank_score is not None and c.rerank_score >= settings.rerank_score_threshold
        ]
    else:
        kept = [
            c for c in candidates if c.dense_score is not None and c.dense_score >= settings.retrieval_score_threshold
        ]
    return kept[: settings.retrieval_list_top_k if list_mode else settings.retrieval_top_k]


def rank_candidates(
    plan: QueryPlan, client: QdrantClient, trace: dict[str, Any] | None = None
) -> tuple[list[Candidate], bool]:
    """Rank every candidate before the evidence gate.

    Args:
        plan: The query plan.
        client: Qdrant client.
        trace: Optional dict that receives timings and diagnostics.

    Returns:
        Candidates in final relevance order, and whether that order comes from the reranker.

    Raises:
        QdrantServiceError: If the vector store is unavailable.
        EmbeddingServiceError: If a local model fails.
    """
    trace = trace if trace is not None else {}
    version = embedding_service.embedding_version()

    t = time.perf_counter()
    dense_vector = embedding_service.embed_query(plan.retrieval_query)
    sparse_vector = (
        embedding_service.embed_sparse_query(plan.retrieval_query) if embedding_service.sparse_enabled() else None
    )
    trace["embed_ms"] = _ms(t)

    t = time.perf_counter()
    results = qdrant_service.query_candidates(
        client, dense_vector, sparse_vector, settings.retrieval_candidates, version, plan.categories
    )
    trace["search_ms"] = _ms(t)
    if not results["dense"]:
        logger.warning("No indexed chunks for the current embedding version; run a sync.")

    candidates = fuse(results, settings.rrf_k)[: settings.retrieval_candidates]

    t = time.perf_counter()
    scores = rerank_service.rerank(plan.retrieval_query, [f"{c.title}\n{c.content}" for c in candidates])
    trace["rerank_ms"] = _ms(t)
    if scores is not None:
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = score
        candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

    trace["retrievers"] = {name: len(points) for name, points in results.items()}
    trace["reranked"] = scores is not None
    return candidates, scores is not None


def fuse(results: dict[str, list[Any]], k: int) -> list[Candidate]:
    """Merge retriever results with Reciprocal Rank Fusion, keyed by chunk.

    Args:
        results: Retriever name to ranked Qdrant points.
        k: RRF constant (higher = flatter rank weighting).

    Returns:
        Unique candidates sorted by fused score.
    """
    by_key: dict[str, Candidate] = {}
    for name, points in results.items():
        for rank, point in enumerate(points, start=1):
            p = point.payload
            # record_id: points written before chunking keep answering until the next sync replaces them.
            entry_id = p.get("entry_id") or p["record_id"]
            key = f"{entry_id}#{p.get('chunk_index', 0)}"
            candidate = by_key.get(key)
            if candidate is None:
                candidate = by_key[key] = Candidate(
                    entry_id,
                    p.get("chunk_index", 0),
                    p["category"],
                    p["title"],
                    p["content"],
                    p.get("url") or "",
                )
            candidate.rrf += 1.0 / (k + rank)
            candidate.ranks[name] = rank
            if name != "sparse":  # dense and category-filtered dense lists share the same cosine
                candidate.dense_score = max(point.score, candidate.dense_score or point.score)
    return sorted(by_key.values(), key=lambda c: c.rrf, reverse=True)


def build_context(candidates: list[Candidate], max_tokens: int) -> tuple[str, list[Candidate], int]:
    """Build numbered, attributed source blocks within a token budget (~4 chars/token).

    The best candidate is always included, even if it alone exceeds the budget.

    Args:
        candidates: Candidates in relevance order.
        max_tokens: Approximate token budget.

    Returns:
        The context text, the candidates included (index = source number - 1), and
        the estimated token count.
    """
    blocks, used, tokens = [], [], 0
    for candidate in candidates:
        header = f"[{len(used) + 1}] {candidate.category.replace('_', ' ')} — {candidate.title}"
        if candidate.url:
            header += f" ({candidate.url})"
        block = f"{header}\n{candidate.content}"
        cost = len(block) // 4 + 1
        if used and tokens + cost > max_tokens:
            break
        blocks.append(block)
        used.append(candidate)
        tokens += cost
    return "\n\n".join(blocks), used, tokens


def apply_citations(answer: str, used: list[Candidate]) -> tuple[str, list[Candidate]]:
    """Normalise and renumber citation markers so `[n]` maps to `sources[n-1]`.

    Markers are renumbered in order of first use; out-of-range ones are dropped;
    grouped `[2, 1]` and gpt-oss `【1】` styles are converted.

    Args:
        answer: Raw LLM answer.
        used: Candidates in the order they were numbered in the context.

    Returns:
        The answer with renumbered markers, and the cited candidates in citation order.
    """
    order: list[int] = []

    def renumber(match: re.Match[str]) -> str:
        valid = [n for n in (int(x) for x in re.split(r"\s*,\s*", match.group(1))) if 1 <= n <= len(used)]
        for n in valid:
            if n not in order:
                order.append(n)
        return "".join(f"[{order.index(n) + 1}]" for n in valid)

    answer = NATIVE_CITATION.sub(r"[\1]", answer)
    return CITATION.sub(renumber, answer), [used[n - 1] for n in order]


def preload_models() -> None:
    """Load all configured models so the first user doesn't pay the loading cost.

    Failures are logged, not raised: models then load on the first request instead.
    """
    try:
        embedding_service.embed_query("warm-up")
        if embedding_service.sparse_enabled():
            embedding_service.embed_sparse_query("warm-up")
        rerank_service.rerank("warm-up", ["warm-up"])
        logger.info("Models preloaded.")
    except Exception:
        logger.warning("Model preload failed; models will load on first request.", exc_info=True)


def _is_refusal(answer: str) -> bool:
    """Return True if the answer is exactly the configured "not available" message."""
    return answer.strip().strip("\"'").rstrip(".") == settings.no_answer_message.strip().rstrip(".")


def _source(c: Candidate) -> dict[str, Any]:
    """Serialise a candidate as an API source."""
    snippet = c.content if len(c.content) <= SNIPPET_CHARS else c.content[: SNIPPET_CHARS - 1].rstrip() + "…"
    return {
        "entry_id": c.entry_id,
        "title": c.title,
        "category": c.category,
        "url": c.url,
        "snippet": snippet,
        "score": _round(c.rerank_score if c.rerank_score is not None else c.dense_score),
    }


def _score(c: Candidate) -> float:
    """The score that decided a candidate's rank: rerank if present, else dense cosine."""
    return c.rerank_score if c.rerank_score is not None else (c.dense_score or 0.0)


def _round(value: float | None) -> float | None:
    """Round a score for output."""
    return round(value, 4) if value is not None else None


def _ms(start: float) -> float:
    """Milliseconds elapsed since `start` (a `time.perf_counter()` value)."""
    return round((time.perf_counter() - start) * 1000, 1)


def _log_trace(plan: QueryPlan, trace: dict[str, Any]) -> None:
    """Log one structured line per request: operational metadata only, never question text or content."""
    summary = {k: v for k, v in trace.items() if k != "candidates"}
    summary.update(
        rewritten=plan.rewritten,
        categories=plan.categories,
        list_mode=plan.list_mode,
        llm=f"groq:{settings.groq_model}",
        embedding=settings.embedding_model,
    )
    logger.info("rag_request %s", json.dumps(summary))
