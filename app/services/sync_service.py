"""Knowledge-base synchronization: Supabase -> chunks -> embeddings -> Qdrant."""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient

from app.models.knowledge import KnowledgeEntryIn
from app.models.sync import SyncReport, SyncStatus
from app.services import chunking, embedding_service, qdrant_service
from app.services.chunking import Chunk
from app.services.supabase_repository import SupabaseRepository

logger = logging.getLogger(__name__)

UPSERT_BATCH = 128


class KnowledgeSyncService:
    """Bring Qdrant in line with Supabase, idempotently.

    Only entries whose content or embedding configuration changed are re-embedded.
    A changed or removed entry has all its points deleted by `entry_id` before the new
    chunks go in, so edits that shrink an entry never leave stale chunks behind.

    ponytail: single-process in-memory lock and status — fine for one instance; move
    both to Redis (or a Qdrant flag) if sync ever runs on several instances.

    Args:
        repository: Where entries are read from.
        client_factory: Returns the Qdrant client to index into.
    """

    def __init__(
        self,
        repository: SupabaseRepository | None = None,
        client_factory: Callable[[], QdrantClient] | None = None,
    ) -> None:
        self.repository = repository or SupabaseRepository()
        self.client_factory = client_factory or qdrant_service.get_client
        self.stage = "idle"
        self.last_report: SyncReport | None = None
        self._lock = asyncio.Lock()

    def status(self) -> SyncStatus:
        """Return the current stage and the last report."""
        return SyncStatus(running=self._lock.locked(), stage=self.stage, last_report=self.last_report)

    async def run(self) -> SyncReport:
        """Run one synchronization.

        Only one runs at a time; a concurrent call returns immediately with an error
        report. Embedding runs in a worker thread so chat keeps being served meanwhile.

        Returns:
            The sync report. Never raises: failures are reported, since callers are
            automation and the admin UI.
        """
        if self._lock.locked():
            return SyncReport(success=False, error="A synchronization is already in progress.")

        async with self._lock:
            started = time.perf_counter()
            try:
                self.stage = "fetching"
                entries = await self.repository.list_entries(active_only=True)
                counts = await asyncio.to_thread(self._index, entries)
                report = SyncReport(success=True, **counts)
                qdrant_service.clear_category_cache()
            except Exception as exc:
                logger.exception("Sync failed")
                report = SyncReport(success=False, error=str(exc))
            finally:
                self.stage = "idle"
            report.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            report.finished_at = datetime.now(UTC)
            self.last_report = report
            logger.info("sync %s", report.model_dump_json(exclude={"error"}))
            return report

    def _index(self, entries: list[KnowledgeEntryIn]) -> dict[str, Any]:
        """Chunk and index entries (blocking CPU and network work)."""
        return index_chunks(self.client_factory(), chunking.build_chunks(entries), on_stage=self._set_stage)

    def _set_stage(self, stage: str) -> None:
        self.stage = stage


def index_chunks(
    client: QdrantClient, chunks: list[Chunk], on_stage: Callable[[str], None] = lambda _: None
) -> dict[str, Any]:
    """Index the full, current set of chunks incrementally.

    Args:
        client: Qdrant client.
        chunks: Chunks of every active entry.
        on_stage: Called with "embedding" and "upserting" as the work progresses.

    Returns:
        Report counts: `total_entries`, `chunks_embedded`, `vectors_upserted`, `added`,
        `updated`, `removed`, `unchanged` (entries) and `reindexed` (bool).
    """
    version = embedding_service.embedding_version()
    reindexed = qdrant_service.ensure_collection(
        client, embedding_service.embedding_dim(), embedding_service.sparse_enabled()
    )
    existing, legacy = ({}, []) if reindexed else qdrant_service.scan_index(client)
    if legacy:
        logger.info("Removing %d points written before chunked indexing.", len(legacy))
        qdrant_service.delete_points(client, legacy)

    by_entry: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_entry.setdefault(chunk.entry_id, []).append(chunk)
    changed = [eid for eid, cs in by_entry.items() if existing.get(eid) != qdrant_service.fingerprint(cs[0], version)]
    removed = [eid for eid in existing if eid not in by_entry]
    to_upsert = [c for eid in changed for c in by_entry[eid]]

    # ponytail: every changed chunk is embedded before anything is upserted, so the
    # admin UI gets clean stages; ~10 MB of vectors per 1k chunks. Interleave
    # embed/upsert per batch if a knowledge base grows to tens of thousands of chunks.
    on_stage("embedding")
    texts = [chunking.embedding_text(c) for c in to_upsert]
    dense = embedding_service.embed_documents(texts)
    sparse = embedding_service.embed_sparse_documents(texts) if texts and embedding_service.sparse_enabled() else None

    on_stage("upserting")
    qdrant_service.delete_entries(client, [eid for eid in changed if eid in existing] + removed)
    for start in range(0, len(to_upsert), UPSERT_BATCH):
        end = start + UPSERT_BATCH
        qdrant_service.upsert_chunks(
            client, to_upsert[start:end], dense[start:end], sparse[start:end] if sparse else None, version
        )

    added = sum(1 for eid in changed if eid not in existing)
    return {
        "total_entries": len(by_entry),
        "chunks_embedded": len(to_upsert),
        "vectors_upserted": len(to_upsert),
        "added": added,
        "updated": len(changed) - added,
        "removed": len(removed),
        "unchanged": len(by_entry) - len(changed),
        "reindexed": reindexed,
    }


knowledge_sync = KnowledgeSyncService()
