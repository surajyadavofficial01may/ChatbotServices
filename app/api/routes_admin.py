"""Admin API behind `/dataentry`: sign-in, schema, knowledge-entry CRUD, bulk import and sync.

Everything except sign-in/out requires the HttpOnly session cookie set by `POST /admin/auth`.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Response, status
from pydantic import BaseModel, Field

from app.api.dependencies import (
    SESSION_COOKIE,
    SESSION_SECONDS,
    check_admin_password,
    issue_session_token,
    login_rate_limit,
    require_admin,
    require_sync_enabled,
)
from app.core.config import settings
from app.models.chat import ErrorResponse
from app.models.knowledge import ENTRY_ID_PATTERN, KnowledgeEntry, KnowledgeEntryIn
from app.models.sync import SyncReport, SyncStatus
from app.services import sync_service
from app.services.supabase_repository import DuplicateEntryError, SupabaseRepository

MAX_BULK_ENTRIES = 2000

router = APIRouter(prefix="/admin", tags=["admin"])
protected = APIRouter(
    dependencies=[Depends(require_admin)],
    responses={401: {"model": ErrorResponse, "description": "Not signed in (no, expired or forged session)."}},
)

EntryId = Annotated[str, Path(pattern=ENTRY_ID_PATTERN)]


class LoginRequest(BaseModel):
    """Admin sign-in."""

    password: str = Field(min_length=1, max_length=1024)


class BulkImportResult(BaseModel):
    """Outcome of a bulk import."""

    success: bool = True
    upserted: int
    duplicates_merged: int = Field(description="Entries in the file that repeated an earlier id (the last one won).")


def get_repository() -> SupabaseRepository:
    """Return the repository the sync service reads from, so CRUD and sync see the same data."""
    return sync_service.knowledge_sync.repository


Repository = Annotated[SupabaseRepository, Depends(get_repository)]


def _cookie_path() -> str:
    return f"{settings.api_prefix}/admin"


@router.post(
    "/auth",
    summary="Sign in",
    description="Checks `ADMIN_PASSWORD` and sets an HttpOnly session cookie (12 h). Limited to 5 attempts/min per IP.",
    dependencies=[Depends(login_rate_limit)],
    responses={
        401: {"model": ErrorResponse, "description": "Wrong password."},
        429: {"model": ErrorResponse, "description": "Too many attempts; see Retry-After."},
        503: {"model": ErrorResponse, "description": "ADMIN_PASSWORD is not configured."},
    },
)
def login(body: LoginRequest, response: Response) -> dict[str, Any]:
    """Verify the admin password and start a session.

    Args:
        body: The password.
        response: Used to set the session cookie.

    Returns:
        `{"success": true, "expires_in": <seconds>}`.
    """
    check_admin_password(body.password)
    response.set_cookie(
        SESSION_COOKIE,
        issue_session_token(),
        max_age=SESSION_SECONDS,
        path=_cookie_path(),
        httponly=True,
        samesite="strict",
        secure=settings.app_env == "production",
    )
    return {"success": True, "expires_in": SESSION_SECONDS}


@router.post("/logout", summary="Sign out")
def logout(response: Response) -> dict[str, bool]:
    """End the session by clearing the cookie.

    Returns:
        `{"success": true}`.
    """
    response.delete_cookie(SESSION_COOKIE, path=_cookie_path(), httponly=True, samesite="strict")
    return {"success": True}


@protected.get("/schema", summary="Knowledge entry schema", description="JSON Schema the admin form renders from.")
def entry_schema() -> dict[str, Any]:
    """Describe the knowledge-entry fields.

    Returns:
        The JSON Schema of `KnowledgeEntryIn`, with `ui` widget hints.
    """
    return KnowledgeEntryIn.model_json_schema()


@protected.get("/entries", summary="List entries", response_model=list[KnowledgeEntry])
async def list_entries(repository: Repository) -> list[KnowledgeEntry]:
    """Return every row in Supabase (all tabs, active or not), ordered by tab and position."""
    return await repository.list_entries()


@protected.post(
    "/entries",
    summary="Create an entry",
    status_code=status.HTTP_201_CREATED,
    response_model=KnowledgeEntry,
    responses={409: {"model": ErrorResponse, "description": "An entry with this ID already exists."}},
)
async def create_entry(entry: KnowledgeEntryIn, repository: Repository) -> KnowledgeEntry:
    """Store a new entry (sync afterwards to make it searchable).

    Raises:
        HTTPException: 409 if the id is taken.
    """
    try:
        return await repository.create_entry(entry)
    except DuplicateEntryError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@protected.put(
    "/entries/{entry_id}",
    summary="Update an entry",
    response_model=KnowledgeEntry,
    responses={404: {"model": ErrorResponse, "description": "No entry with this ID."}},
)
async def update_entry(entry_id: EntryId, entry: KnowledgeEntryIn, repository: Repository) -> KnowledgeEntry:
    """Replace an entry's fields; its id never changes.

    Raises:
        HTTPException: 404 if the entry doesn't exist.
    """
    updated = await repository.update_entry(entry_id, entry)
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No entry with this ID.")
    return updated


@protected.delete(
    "/entries/{entry_id}",
    summary="Delete an entry",
    responses={404: {"model": ErrorResponse, "description": "No entry with this ID."}},
)
async def delete_entry(entry_id: EntryId, repository: Repository) -> dict[str, bool]:
    """Delete an entry from Supabase (the next sync removes it from Qdrant).

    Raises:
        HTTPException: 404 if the entry doesn't exist.
    """
    if not await repository.delete_entry(entry_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No entry with this ID.")
    return {"success": True}


@protected.post(
    "/entries/bulk",
    summary="Bulk import",
    description=(
        f"Insert or update up to {MAX_BULK_ENTRIES} entries by id in one request. Send a JSON array of entries. "
        'Rows copied from the per-tab views or a spreadsheet (active as "TRUE"/"FALSE") import unchanged.'
    ),
    response_model=BulkImportResult,
)
async def bulk_import(
    entries: Annotated[list[KnowledgeEntryIn], Body(max_length=MAX_BULK_ENTRIES)], repository: Repository
) -> BulkImportResult:
    """Upsert many entries; a repeated id within the file keeps its last occurrence."""
    unique = list({entry.id: entry for entry in entries}.values())
    upserted = await repository.upsert_entries(unique)
    return BulkImportResult(upserted=upserted, duplicates_merged=len(entries) - len(unique))


@protected.post(
    "/sync",
    summary="Sync to Qdrant",
    description="Same operation as the webhook. Poll `GET /admin/sync` meanwhile for the current stage.",
    response_model=SyncReport,
    dependencies=[Depends(require_sync_enabled)],
    responses={403: {"model": ErrorResponse, "description": "SYNC_ENABLED is false."}},
)
async def admin_sync() -> SyncReport:
    """Synchronize Supabase -> Qdrant now.

    Returns:
        The sync report (`success: false` with an error if the sync failed).
    """
    return await sync_service.knowledge_sync.run()


@protected.get("/sync", summary="Sync status", response_model=SyncStatus)
def sync_status() -> SyncStatus:
    """Return whether a sync is running, its stage, and the last report since startup."""
    return sync_service.knowledge_sync.status()


router.include_router(protected)
