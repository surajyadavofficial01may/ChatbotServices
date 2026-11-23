"""Knowledge-base synchronization: Supabase -> documents -> embeddings -> Qdrant."""

import asyncio
import logging
from typing import Any

from qdrant_client import QdrantClient

from app.models.sync import SyncReport
from app.services import data_fetcher, embedding_service, qdrant_service
from app.services.data_fetcher import Document, embedding_text

logger = logging.getLogger(__name__)

EMBED_BATCH = 64

# ponytail: single-process in-memory lock — fine for one Render instance,
# move to a Qdrant/Redis-backed lock if sync ever runs on multiple instances.
_sync_lock = asyncio.Lock()


async def run_sync() -> SyncReport:
    """Pull the knowledge base from Supabase and bring Qdrant up to date.

    Only one sync runs at a time; a concurrent call returns immediately with an
    error report. The blocking work runs in a worker thread so chat requests keep
    being served meanwhile.

    Returns:
        A report of added/updated/removed/unchanged records, or the failure reason.
        Never raises: failures are reported, since callers are automation.
    """
    if _sync_lock.locked():
        return SyncReport(success=False, error="A synchronization is already in progress.")

    async with _sync_lock:
        try:
            counts = await asyncio.to_thread(_sync_blocking)
            return SyncReport(success=True, **counts)
        except Exception as exc:
            logger.exception("Sync failed")
            return SyncReport(success=False, error=str(exc))


def _sync_blocking() -> dict[str, Any]:
    """Fetch, validate and index the knowledge base (blocking I/O and CPU work)."""
    records = data_fetcher.fetch_knowledge_records()
    documents = data_fetcher.build_documents(records)
    return index_documents(qdrant_service.get_client(), documents)


def index_documents(client: QdrantClient, documents: list[Document]) -> dict[str, Any]:
    """Index documents incrementally and idempotently.

    Only records whose content or embedding configuration changed are re-embedded;
    records no longer in the source are deleted; a schema mismatch recreates the
    collection and re-indexes everything.

    Args:
        client: Qdrant client.
        documents: The full, current set of documents.

    Returns:
        Counts: `added`, `updated`, `removed`, `unchanged`, and `reindexed` (bool).
    """
    version = embedding_service.embedding_version()
    reindexed = qdrant_service.ensure_collection(
        client, embedding_service.embedding_dim(), embedding_service.sparse_enabled()
    )
    existing = {} if reindexed else qdrant_service.fetch_existing_fingerprints(client)

    current_ids = {d.id for d in documents}
    to_upsert = [d for d in documents if existing.get(d.id) != qdrant_service.fingerprint(d, version)]
    to_remove = [rid for rid in existing if rid not in current_ids]

    for start in range(0, len(to_upsert), EMBED_BATCH):
        batch = to_upsert[start : start + EMBED_BATCH]
        texts = [embedding_text(d) for d in batch]
        sparse = embedding_service.embed_sparse_documents(texts) if embedding_service.sparse_enabled() else None
        qdrant_service.upsert_documents(client, batch, embedding_service.embed_documents(texts), sparse, version)
    qdrant_service.delete_by_record_ids(client, to_remove)

    added = sum(1 for d in to_upsert if d.id not in existing)
    return {
        "added": added,
        "updated": len(to_upsert) - added,
        "removed": len(to_remove),
        "unchanged": len(documents) - len(to_upsert),
        "reindexed": reindexed,
    }
