"""Optional cross-encoder reranking (FastEmbed), with graceful fallback."""

import logging

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.core.config import settings
from app.services.embedding_service import load_model

logger = logging.getLogger(__name__)


def rerank_enabled() -> bool:
    """Return True when a reranker model is configured."""
    return bool(settings.reranker_model)


def rerank(query: str, passages: list[str]) -> list[float] | None:
    """Score each passage's relevance to the query with a cross-encoder.

    Args:
        query: The search query.
        passages: Candidate passages to score.

    Returns:
        One score per passage (higher = more relevant), or None when reranking is
        disabled or the model fails — callers then keep the fusion order instead of
        failing the request.
    """
    if not rerank_enabled() or not passages:
        return None
    name = settings.reranker_model
    try:
        model = load_model("rerank", name, lambda: TextCrossEncoder(model_name=name))
        return [float(s) for s in model.rerank(query, passages)]
    except Exception:
        logger.warning("Reranker '%s' unavailable; falling back to fusion order.", name, exc_info=True)
        return None
