"""Qdrant vector store: collection schema, chunk upserts/deletes, category facets and batched hybrid queries."""

import logging
import time
import uuid
from collections import defaultdict
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.services.chunking import Chunk

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"
# Every payload field used in a query or delete filter. Qdrant Cloud runs in strict
# mode, which rejects filters on unindexed fields.
FILTERED_FIELDS = ("category", "embedding_version", "entry_id")
CATEGORY_CACHE_SECONDS = 300


class QdrantServiceError(ExternalServiceError):
    """Qdrant is unconfigured, unreachable, or rejected a request."""


def point_id(entry_id: str, chunk_index: int) -> str:
    """Derive a stable Qdrant point id for one chunk of an entry.

    The same chunk always maps to the same UUID5, so re-upserts overwrite it instead
    of duplicating it.

    Args:
        entry_id: Source entry id.
        chunk_index: Chunk position within the entry.

    Returns:
        A UUID string.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{entry_id}#{chunk_index}"))


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return the shared Qdrant Cloud client.

    Returns:
        A client configured from `QDRANT_URL` / `QDRANT_API_KEY`.

    Raises:
        QdrantServiceError: If `QDRANT_URL` is not set.
    """
    if not settings.qdrant_url:
        raise QdrantServiceError("QDRANT_URL is not configured.")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=10)


def ensure_collection(client: QdrantClient, dense_dim: int, with_sparse: bool) -> bool:
    """Create the collection, or recreate it if its schema no longer matches.

    Qdrant is a derived index (Supabase is the source of truth), so the caller
    re-indexes everything after a recreate.

    Args:
        client: Qdrant client.
        dense_dim: Size of the dense vectors.
        with_sparse: Whether a BM25 sparse vector is required.

    Returns:
        True if the collection was created or recreated, False if it already matched.
    """
    name = settings.qdrant_collection_name
    if client.collection_exists(name):
        info = client.get_collection(name)
        if _schema_matches(info, dense_dim, with_sparse):
            _ensure_payload_indexes(client, set((info.payload_schema or {}).keys()))
            return False
        logger.warning("Collection '%s' does not match the configured models; recreating it.", name)
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: qmodels.VectorParams(size=dense_dim, distance=qmodels.Distance.COSINE)},
        sparse_vectors_config=(
            {SPARSE: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)} if with_sparse else None
        ),
    )
    _ensure_payload_indexes(client, set())
    return True


def _ensure_payload_indexes(client: QdrantClient, existing: set[str]) -> None:
    """Create any missing keyword index on the fields filters use."""
    for field in FILTERED_FIELDS:
        if field not in existing:
            logger.info("Creating payload index on '%s'.", field)
            client.create_payload_index(
                settings.qdrant_collection_name, field_name=field, field_schema=qmodels.PayloadSchemaType.KEYWORD
            )


def _schema_matches(info: Any, dense_dim: int, with_sparse: bool) -> bool:
    """Return True if an existing collection has the expected named vectors."""
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or DENSE not in vectors or vectors[DENSE].size != dense_dim:
        return False
    return not with_sparse or SPARSE in (info.config.params.sparse_vectors or {})


def fingerprint(chunk: Chunk, version: str) -> str:
    """Identify an entry's indexed state: embedding version plus entry hash.

    Args:
        chunk: Any chunk of the entry (they share the entry hash).
        version: Current embedding version.

    Returns:
        A string that changes whenever the entry's points must be rebuilt.
    """
    return f"{version}#{chunk.entry_hash}"


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[qmodels.SparseVector] | None,
    version: str,
) -> None:
    """Insert or overwrite one point per chunk.

    Args:
        client: Qdrant client.
        chunks: Chunks to store.
        dense_vectors: One dense vector per chunk.
        sparse_vectors: One BM25 vector per chunk, or None when hybrid search is off.
        version: Embedding version stamped on each point.
    """
    points = []
    for i, chunk in enumerate(chunks):
        vector: dict[str, Any] = {DENSE: dense_vectors[i]}
        if sparse_vectors is not None:
            vector[SPARSE] = sparse_vectors[i]
        points.append(
            qmodels.PointStruct(
                id=point_id(chunk.entry_id, chunk.chunk_index),
                vector=vector,
                payload={
                    "entry_id": chunk.entry_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_count": chunk.chunk_count,
                    "category": chunk.category,
                    "title": chunk.title,
                    "content": chunk.text,
                    "url": chunk.url,
                    "fingerprint": fingerprint(chunk, version),
                    "embedding_version": version,
                },
            )
        )
    if points:
        client.upsert(collection_name=settings.qdrant_collection_name, points=points)


def delete_entries(client: QdrantClient, entry_ids: list[str]) -> None:
    """Delete every point (all chunks) of the given entries.

    Args:
        client: Qdrant client.
        entry_ids: Entry ids to remove.
    """
    if not entry_ids:
        return
    client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(must=[qmodels.FieldCondition(key="entry_id", match=qmodels.MatchAny(any=entry_ids))])
        ),
    )


def scan_index(client: QdrantClient) -> tuple[dict[str, str], list[Any]]:
    """Read every indexed entry's fingerprint, for sync diffing.

    An entry whose chunks disagree on the fingerprint, or that is missing chunks (an
    interrupted sync), gets an empty fingerprint so the next sync rebuilds it. Points
    without an `entry_id` were written before chunked indexing (one point per record,
    different ids); they are returned separately so the sync can delete them.

    Args:
        client: Qdrant client.

    Returns:
        Mapping of entry id to stored fingerprint ("" when its points are inconsistent),
        and the ids of legacy points to delete.
    """
    seen: dict[str, list[tuple[str, int]]] = defaultdict(list)
    legacy: list[Any] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection_name,
            with_payload=["entry_id", "fingerprint", "chunk_count"],
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        for p in points:
            payload = p.payload or {}
            if "entry_id" not in payload:
                legacy.append(p.id)
                continue
            seen[payload["entry_id"]].append((payload.get("fingerprint", ""), payload.get("chunk_count", 1)))
        if next_offset is None:
            break
    fingerprints = {
        entry_id: chunks[0][0] if len({fp for fp, _ in chunks}) == 1 and len(chunks) == chunks[0][1] else ""
        for entry_id, chunks in seen.items()
    }
    return fingerprints, legacy


def delete_points(client: QdrantClient, point_ids: list[Any]) -> None:
    """Delete points by id.

    Args:
        client: Qdrant client.
        point_ids: Point ids to remove.
    """
    if point_ids:
        client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=qmodels.PointIdsList(points=point_ids),
        )


_category_cache: dict[str, Any] = {"at": float("-inf"), "values": []}


def list_categories(client: QdrantClient, version: str) -> list[str]:
    """Return the categories present in the index (cached for a few minutes).

    Used for "list all X" routing, which is only a retrieval boost, so a Qdrant
    failure here returns the last known list instead of failing the request.

    Args:
        client: Qdrant client.
        version: Only categories of points with this embedding version count.

    Returns:
        Distinct category names.
    """
    now = time.monotonic()
    if now - _category_cache["at"] > CATEGORY_CACHE_SECONDS:
        try:
            response = client.facet(
                settings.qdrant_collection_name, key="category", facet_filter=_filter(version), limit=200
            )
            _category_cache["values"] = [str(hit.value) for hit in response.hits]
        except Exception:
            logger.warning("Could not list categories; keeping the previous list.", exc_info=True)
        _category_cache["at"] = now
    return _category_cache["values"]


def clear_category_cache() -> None:
    """Forget the cached category list (after a sync, or between tests)."""
    _category_cache.update(at=float("-inf"), values=[])


def _filter(version: str, category: str | None = None) -> qmodels.Filter:
    """Build the payload filter: current embedding version, optionally one category."""
    must = [qmodels.FieldCondition(key="embedding_version", match=qmodels.MatchValue(value=version))]
    if category:
        must.append(qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value=category)))
    return qmodels.Filter(must=must)


def query_candidates(
    client: QdrantClient,
    dense_vector: list[float],
    sparse_vector: qmodels.SparseVector | None,
    limit: int,
    version: str,
    categories: list[str],
) -> dict[str, list[qmodels.ScoredPoint]]:
    """Run every retriever in one round-trip.

    Dense search, BM25 search (skipped if the query has no known terms) and one
    category-filtered dense search per routed category.

    Args:
        client: Qdrant client.
        dense_vector: Query dense vector.
        sparse_vector: Query BM25 vector, or None for dense-only retrieval.
        limit: Candidates per retriever.
        version: Only points with this embedding version are matched.
        categories: Categories to add a filtered retriever for.

    Returns:
        Retriever name ("dense", "sparse", "category:<name>") to its ranked hits.

    Raises:
        QdrantServiceError: If Qdrant is unreachable or rejects the query (e.g. the
            collection is missing — run a sync).
    """
    names = ["dense"]
    requests = [
        qmodels.QueryRequest(query=dense_vector, using=DENSE, limit=limit, filter=_filter(version), with_payload=True)
    ]
    if sparse_vector is not None and sparse_vector.indices:
        names.append("sparse")
        requests.append(
            qmodels.QueryRequest(
                query=sparse_vector, using=SPARSE, limit=limit, filter=_filter(version), with_payload=True
            )
        )
    for category in categories:
        names.append(f"category:{category}")
        requests.append(
            qmodels.QueryRequest(
                query=dense_vector, using=DENSE, limit=limit, filter=_filter(version, category), with_payload=True
            )
        )
    try:
        responses = client.query_batch_points(settings.qdrant_collection_name, requests=requests)
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        raise QdrantServiceError(
            f"Qdrant query failed ({type(exc).__name__}). If the collection is missing or predates this version, "
            f"run a sync to rebuild it. Detail: {exc}"
        ) from exc
    return {name: response.points for name, response in zip(names, responses, strict=True)}
