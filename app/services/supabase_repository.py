"""Supabase knowledge-entry storage over Supabase's REST API (PostgREST), with async httpx.

The service-role key bypasses Row Level Security, so it must only ever live on the server.
"""

from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.models.knowledge import KnowledgeEntry, KnowledgeEntryIn

PAGE_SIZE = 1000  # Supabase's default max rows per request


class SupabaseServiceError(ExternalServiceError):
    """Supabase is unconfigured, unreachable, or rejected the request.

    Messages are built here from status codes and PostgREST's own text only, so they
    are returned to the (admin) client to make setup problems fixable.
    """

    expose = True


class DuplicateEntryError(Exception):
    """An entry with this id already exists."""


class SupabaseRepository:
    """Async CRUD for knowledge entries.

    Settings are read on every call, so configuration changes (and test overrides)
    apply without rebuilding the repository.

    Args:
        transport: Optional httpx transport, for tests.
    """

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def list_entries(self, active_only: bool = False) -> list[KnowledgeEntry]:
        """Return every entry, ordered by tab and position.

        Args:
            active_only: Only rows with `active = true` (what the sync indexes).

        Returns:
            All matching entries, paging past Supabase's per-request row cap.
        """
        params: dict[str, Any] = {"select": "*", "order": "category.asc,sort_order.asc,id.asc", "limit": PAGE_SIZE}
        if active_only:
            params["active"] = "eq.true"
        rows: list[dict[str, Any]] = []
        while True:
            page = await self._request("GET", params={**params, "offset": len(rows)})
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return [KnowledgeEntry.model_validate(row) for row in rows]

    async def create_entry(self, entry: KnowledgeEntryIn) -> KnowledgeEntry:
        """Insert a new entry.

        Args:
            entry: The entry to store.

        Returns:
            The stored entry, with timestamps.

        Raises:
            DuplicateEntryError: If the id is taken.
        """
        rows = await self._request("POST", json=_row(entry), prefer="return=representation")
        return KnowledgeEntry.model_validate(rows[0])

    async def update_entry(self, entry_id: str, entry: KnowledgeEntryIn) -> KnowledgeEntry | None:
        """Replace an entry's fields (its id never changes).

        Args:
            entry_id: Id of the entry to update.
            entry: The new field values.

        Returns:
            The updated entry, or None if no entry has that id.
        """
        row = {k: v for k, v in _row(entry).items() if k != "id"}
        rows = await self._request("PATCH", params={"id": f"eq.{entry_id}"}, json=row, prefer="return=representation")
        return KnowledgeEntry.model_validate(rows[0]) if rows else None

    async def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry.

        Args:
            entry_id: Id of the entry to delete.

        Returns:
            True if it existed.
        """
        rows = await self._request("DELETE", params={"id": f"eq.{entry_id}"}, prefer="return=representation")
        return bool(rows)

    async def upsert_entries(self, entries: list[KnowledgeEntryIn]) -> int:
        """Insert or update many entries by id, in one request.

        Args:
            entries: Entries with unique ids.

        Returns:
            How many rows were written.
        """
        if not entries:
            return 0
        rows = await self._request(
            "POST",
            json=[_row(e) for e in entries],
            prefer="resolution=merge-duplicates,return=representation",
        )
        return len(rows)

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        json: Any = None,
        prefer: str | None = None,
    ) -> Any:
        """Call the table's REST endpoint and decode the JSON reply.

        Error messages carry the status and PostgREST's own message, never keys or headers.

        Raises:
            SupabaseServiceError: Unconfigured, unreachable, or any non-2xx reply except 409.
            DuplicateEntryError: On 409 (unique violation).
        """
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise SupabaseServiceError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not configured.")
        key = settings.supabase_service_role_key
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        if prefer:
            headers["Prefer"] = prefer
        url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{settings.supabase_table_name}"
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=30) as client:
                response = await client.request(method, url, params=params, json=json, headers=headers)
        except httpx.HTTPError as exc:
            raise SupabaseServiceError(f"Supabase request failed ({type(exc).__name__}).") from exc

        if response.status_code == 409:
            raise DuplicateEntryError("An entry with this ID already exists.")
        if response.status_code >= 400:
            raise SupabaseServiceError(_error_message(response))
        return response.json() if response.content else []


def _row(entry: KnowledgeEntryIn) -> dict[str, Any]:
    """Serialise an entry as a table row (writable columns only)."""
    return entry.model_dump(mode="json", include=set(KnowledgeEntryIn.model_fields))


def _error_message(response: httpx.Response) -> str:
    """Turn a PostgREST error into an actionable message."""
    try:
        body = response.json()
        detail = body.get("message") or body.get("hint") or ""
        code = body.get("code", "")
    except (ValueError, AttributeError):
        detail, code = "", ""
    if code in {"42P01", "PGRST205"}:  # undefined table / not in the schema cache
        return (
            f"Table '{settings.supabase_table_name}' not found. "
            "Run supabase/schema.sql in the Supabase SQL Editor (see README)."
        )
    if response.status_code in {401, 403}:
        return f"Supabase rejected the key (HTTP {response.status_code}). Use the service_role / secret key."
    return f"Supabase returned HTTP {response.status_code}{': ' + detail if detail else ''}."
