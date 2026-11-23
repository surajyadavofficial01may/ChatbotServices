"""Qdrant vector store: collection schema, upserts/deletes, and batched hybrid queries."""

import logging
import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.services.data_fetcher import Document

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"
FILTERED_FIELDS = ("category", "embedding_version")  # every payload field used in a query filter


class QdrantServiceError(ExternalServiceError):
    """Qdrant is unconfigured, unreachable, or rejected a request."""


def deterministic_point_id(record_id: str) -> str:
    """Derive a stable Qdrant point id from a record id.

    The same record always maps to the same UUID5, so repeated syncs overwrite the
    point instead of duplicating it.

    Args:
        record_id: Source record id.

    Returns:
        A UUID string.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))


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
    """Create the collection, or recreate it if its vector schema no longer matches.

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
    """Create any missing keyword index on the fields queries filter by.

    Qdrant Cloud runs in strict mode, which rejects filters on unindexed fields, so a
    collection that has the right vectors but lacks an index must still be repaired.
    """
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


def fingerprint(doc: Document, version: str) -> str:
    """Identify a document's indexed state: embedding version plus content hash.

    Args:
        doc: The document.
        version: Current embedding version.

    Returns:
        A string that changes whenever the stored point must be rebuilt.
    """
    return f"{version}#{doc.content_hash}"


def upsert_documents(
    client: QdrantClient,
    documents: list[Document],
    dense_vectors: list[list[float]],
    sparse_vectors: list[qmodels.SparseVector] | None,
    version: str,
) -> None:
    """Insert or overwrite points; the point id is derived from the record id, so re-upserts never duplicate.

    Args:
        client: Qdrant client.
        documents: Documents to store.
        dense_vectors: One dense vector per document.
        sparse_vectors: One BM25 vector per document, or None when hybrid search is off.
        version: Embedding version stamped on each point.
    """
    points = []
    for i, doc in enumerate(documents):
        vector: dict[str, Any] = {DENSE: dense_vectors[i]}
        if sparse_vectors is not None:
            vector[SPARSE] = sparse_vectors[i]
        points.append(
            qmodels.PointStruct(
                id=deterministic_point_id(doc.id),
                vector=vector,
                payload={
                    "record_id": doc.id,
                    "category": doc.category,
                    "title": doc.title,
                    "content": doc.content,
                    "url": doc.url,
                    "fingerprint": fingerprint(doc, version),
                    "embedding_version": version,
                },
            )
        )
    if points:
        client.upsert(collection_name=settings.qdrant_collection_name, points=points)


def delete_by_record_ids(client: QdrantClient, record_ids: list[str]) -> None:
    """Delete the points belonging to the given record ids.

    Args:
        client: Qdrant client.
        record_ids: Source record ids to remove.
    """
    if not record_ids:
        return
    client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=qmodels.PointIdsList(points=[deterministic_point_id(rid) for rid in record_ids]),
    )


def fetch_existing_fingerprints(client: QdrantClient) -> dict[str, str]:
    """Read the fingerprint of every stored point, for sync diffing.

    Args:
        client: Qdrant client.

    Returns:
        Mapping of record id to stored fingerprint.
    """
    result: dict[str, str] = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection_name,
            with_payload=["record_id", "fingerprint"],
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        for p in points:
            result[p.payload["record_id"]] = p.payload.get("fingerprint", "")
        if next_offset is None:
            break
    return result


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
            collection is missing or still has the pre-hybrid schema — run a sync).
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
            f"Qdrant query failed ({type(exc).__name__}). If the collection predates this version or "
            f"is missing indexes, run a sync (POST /api/v1/webhook/sync) to rebuild it. Detail: {exc}"
        ) from exc
    return {name: response.points for name, response in zip(names, responses, strict=True)}
