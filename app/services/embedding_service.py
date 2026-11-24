"""Local embedding models (FastEmbed): dense vectors, BM25 sparse vectors, model loading."""

import logging
import threading
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

# Bump whenever chunking.embedding_text changes shape, so the next sync
# re-embeds every entry instead of mixing old and new text formats.
TEXT_TEMPLATE_VERSION = "2"

# fastembed's query_embed() applies no instruction prefix, so models trained with
# one need it supplied here (from each model card). Unlisted models get none.
_BGE_QUERY = "Represent this sentence for searching relevant passages: "
MODEL_PREFIXES: dict[str, tuple[str, str]] = {  # model -> (query prefix, document prefix)
    "baai/bge-small-en-v1.5": (_BGE_QUERY, ""),
    "baai/bge-base-en-v1.5": (_BGE_QUERY, ""),
    "snowflake/snowflake-arctic-embed-xs": (_BGE_QUERY, ""),
    "snowflake/snowflake-arctic-embed-s": (_BGE_QUERY, ""),
    "snowflake/snowflake-arctic-embed-m": (_BGE_QUERY, ""),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "nomic-ai/nomic-embed-text-v1.5-q": ("search_query: ", "search_document: "),
}


class EmbeddingServiceError(ExternalServiceError):
    """A local model could not be loaded or produced invalid output."""


_models: dict[tuple[str, str], Any] = {}
_models_lock = threading.Lock()


def load_model(kind: str, name: str, factory: Callable[[], Any]) -> Any:
    """Load a model once per process, even under concurrent first requests.

    Args:
        kind: Model family used as part of the cache key ("dense", "sparse", "rerank").
        name: Model name, the other part of the cache key.
        factory: Zero-argument callable that builds the model.

    Returns:
        The cached model instance.

    Raises:
        EmbeddingServiceError: If the factory fails (unknown model, download error...).
    """
    key = (kind, name)
    with _models_lock:
        if key not in _models:
            logger.info("Loading %s model %s", kind, name)
            try:
                _models[key] = factory()
            except Exception as exc:
                raise EmbeddingServiceError(f"Failed to load {kind} model '{name}': {exc}") from exc
        return _models[key]


def _dense() -> TextEmbedding:
    """Return the configured dense embedding model."""
    name = settings.embedding_model
    return load_model("dense", name, lambda: TextEmbedding(model_name=name))


def _prefixes() -> tuple[str, str]:
    """Return the (query, document) prefixes for the configured dense model."""
    query_prefix, doc_prefix = MODEL_PREFIXES.get(settings.embedding_model.lower(), ("", ""))
    if settings.embedding_query_prefix is not None:
        query_prefix = settings.embedding_query_prefix
    return query_prefix, doc_prefix


@lru_cache(maxsize=8)
def _known_dim(model_name: str) -> int | None:
    """Look up a model's vector size in FastEmbed's registry (None if unlisted)."""
    return next(
        (m["dim"] for m in TextEmbedding.list_supported_models() if m["model"].lower() == model_name.lower()), None
    )


def embedding_dim() -> int:
    """Return the dense vector size of the configured model.

    Returns:
        Dimension from the model registry, or measured from one embedding for custom models.
    """
    return _known_dim(settings.embedding_model) or len(embed_query("dimension probe"))


def embedding_version() -> str:
    """Identify everything that shapes a stored vector.

    Stored on every Qdrant point and used to filter queries, so vectors produced by a
    different model, prefix or text template are never mixed with current ones.

    Returns:
        A string combining dense model, sparse model, text template and prefixes.
    """
    query_prefix, doc_prefix = _prefixes()
    return (
        f"{settings.embedding_model}|{settings.sparse_model or 'no-sparse'}"
        f"|t{TEXT_TEMPLATE_VERSION}|q:{query_prefix}|d:{doc_prefix}"
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed documents with the dense model.

    Args:
        texts: Texts to embed (the document prefix is added here).

    Returns:
        One vector per input text, in order.

    Raises:
        EmbeddingServiceError: If the model can't load or returns the wrong dimension.
    """
    if not texts:
        return []
    _, doc_prefix = _prefixes()
    vectors = [vec.tolist() for vec in _dense().embed([doc_prefix + t for t in texts])]
    _validate_dims(vectors)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a search query with the dense model (cached per model and text).

    Args:
        text: The query (the model's query instruction is added here).

    Returns:
        The query vector.

    Raises:
        EmbeddingServiceError: If the model can't load or returns the wrong dimension.
    """
    query_prefix, _ = _prefixes()
    return list(_cached_query(settings.embedding_model, query_prefix + text))


@lru_cache(maxsize=512)
def _cached_query(model_name: str, text: str) -> tuple[float, ...]:
    """Embed a fully prefixed query; `model_name` is part of the cache key only."""
    vector = next(iter(_dense().query_embed(text))).tolist()
    _validate_dims([vector])
    return tuple(vector)


def _validate_dims(vectors: list[list[float]]) -> None:
    """Raise if any vector doesn't match the configured model's known dimension."""
    expected = _known_dim(settings.embedding_model)
    if expected and any(len(v) != expected for v in vectors):
        raise EmbeddingServiceError(f"Embedding dimension mismatch: expected {expected}.")


def sparse_enabled() -> bool:
    """Return True when hybrid (BM25) retrieval is configured."""
    return bool(settings.sparse_model)


def _sparse() -> SparseTextEmbedding:
    """Return the configured sparse (BM25) model."""
    name = settings.sparse_model
    return load_model("sparse", name, lambda: SparseTextEmbedding(model_name=name))


def embed_sparse_documents(texts: list[str]) -> list[qmodels.SparseVector]:
    """Encode documents as BM25 sparse vectors.

    Args:
        texts: Texts to encode.

    Returns:
        One sparse vector per input text, in order.
    """
    return [_to_sparse(e) for e in _sparse().embed(texts)]


def embed_sparse_query(text: str) -> qmodels.SparseVector:
    """Encode a query as a BM25 sparse vector.

    Args:
        text: The query.

    Returns:
        The sparse query vector (empty if no term is known).
    """
    return _to_sparse(next(iter(_sparse().query_embed(text))))


def _to_sparse(embedding: Any) -> qmodels.SparseVector:
    """Convert a FastEmbed sparse embedding into a Qdrant SparseVector."""
    return qmodels.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
