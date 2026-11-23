"""Knowledge-base synchronization schemas."""

from pydantic import BaseModel, Field


class SyncReport(BaseModel):
    """Outcome of a knowledge-base synchronization."""

    success: bool
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    reindexed: bool = Field(default=False, description="Collection was (re)created, e.g. after a model change.")
    error: str | None = None
