"""Deterministic retrieval and generation metrics. Binary relevance throughout."""

import math
import re


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return 1.0 if any relevant id is in the top k, else 0.0.

    Args:
        ranked: Retrieved ids, best first.
        relevant: Ground-truth relevant ids.
        k: Cut-off.

    Returns:
        1.0 or 0.0.
    """
    return 1.0 if any(doc_id in relevant for doc_id in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return the share of relevant ids found in the top k.

    Args:
        ranked: Retrieved ids, best first.
        relevant: Ground-truth relevant ids (non-empty).
        k: Cut-off.

    Returns:
        Recall between 0 and 1.
    """
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return the share of the top k that is relevant (over what was actually returned).

    Args:
        ranked: Retrieved ids, best first.
        relevant: Ground-truth relevant ids.
        k: Cut-off.

    Returns:
        Precision between 0 and 1 (0 when nothing was returned).
    """
    top = ranked[:k]
    return len(set(top) & relevant) / len(top) if top else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """Return 1 / rank of the first relevant id (0 if none).

    Args:
        ranked: Retrieved ids, best first.
        relevant: Ground-truth relevant ids.

    Returns:
        Reciprocal rank between 0 and 1.
    """
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return normalised discounted cumulative gain at k with binary relevance.

    Args:
        ranked: Retrieved ids, best first.
        relevant: Ground-truth relevant ids.
        k: Cut-off.

    Returns:
        nDCG between 0 and 1.
    """
    dcg = sum(1.0 / math.log2(i + 1) for i, doc_id in enumerate(ranked[:k], start=1) if doc_id in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_scores(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Compute all ranking metrics for one query.

    Args:
        ranked: Retrieved ids, best first.
        relevant: Ground-truth relevant ids.
        k: Cut-off.

    Returns:
        Hit@k, Recall@k, Precision@k, MRR and nDCG@k keyed by name.
    """
    return {
        f"hit@{k}": hit_at_k(ranked, relevant, k),
        f"recall@{k}": recall_at_k(ranked, relevant, k),
        f"precision@{k}": precision_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant),
        f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
    }


def mean(values: list[float]) -> float:
    """Return the arithmetic mean (0.0 for an empty list)."""
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    """Return the nearest-rank percentile.

    Args:
        values: Samples.
        pct: Percentile between 0 and 100.

    Returns:
        The percentile value (0.0 for an empty list).
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


DECLINE = re.compile(r"\b(can(?:'|’)?t|cannot|won(?:'|’)?t) (comply|help with that|assist with that)", re.I)


def is_refusal(answer: str, no_answer_message: str | None = None) -> bool:
    """Return True for a pure "not available" reply or an explicit decline.

    Both mean nothing was answered from outside the knowledge base. Length-bounded
    so a partial answer that also notes a gap isn't counted as a refusal.

    Args:
        answer: Generated answer.
        no_answer_message: The refusal sentence (defaults to `NO_ANSWER_MESSAGE`).

    Returns:
        Whether the answer is a refusal.
    """
    if no_answer_message is None:
        from app.core.config import settings

        no_answer_message = settings.no_answer_message
    marker = no_answer_message.strip().rstrip(".").lower()
    return len(answer) < 250 and (marker in answer.lower() or bool(DECLINE.search(answer)))


def fact_recall(answer: str, must_include: list[str]) -> float:
    """Return the share of expected facts present in the answer.

    Matching ignores case and whitespace, so "81 %" or "Acme Notes" written with a
    narrow no-break space (common in LLM output) still count.

    Args:
        answer: Generated answer.
        must_include: Expected facts.

    Returns:
        Fact recall between 0 and 1 (1.0 when nothing is expected).
    """
    if not must_include:
        return 1.0
    squashed = re.sub(r"\s+", "", answer.lower())
    return sum(1 for fact in must_include if re.sub(r"\s+", "", fact.lower()) in squashed) / len(must_include)


def cited_indices(answer: str) -> list[int]:
    """Return the source numbers cited as `[n]` in an answer, in order of appearance."""
    return [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
