"""Knowledge-base synchronization schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SyncReport(BaseModel):
    """Outcome of a Supabase -> Qdrant synchronization.

    `added` / `updated` / `removed` / `unchanged` count entries; `chunks_embedded` and
    `vectors_upserted` count chunks (each chunk becomes exactly one vector).
    """

    success: bool
    total_entries: int = Field(default=0, description="Active entries read from Supabase.")
    chunks_embedded: int = 0
    vectors_upserted: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    reindexed: bool = Field(default=False, description="Collection was (re)created, e.g. after a model change.")
    duration_ms: float = 0.0
    finished_at: datetime | None = None
    error: str | None = None


class SyncStatus(BaseModel):
    """What the sync is doing right now, and how the last one went (since the server started)."""

    running: bool
    stage: Literal["idle", "fetching", "embedding", "upserting"]
    last_report: SyncReport | None = None
