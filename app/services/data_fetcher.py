"""Knowledge-base ingestion source: fetch from Supabase, validate, and build documents."""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)

BUCKET = "portfolio-knowledge-base"
KNOWLEDGE_FILE = "Portfolio_Knowledge_Base.json"

REQUIRED_CATEGORIES = [
    "Personal_Details",
    "Work_Experience",
    "Technical_Skills",
    "Projects",
    "Articles",
    "Social_Links",
    "FAQs",
]


class SupabaseServiceError(ExternalServiceError):
    """Supabase is unconfigured, unreachable, or returned an unexpected file."""


class KnowledgeValidationError(Exception):
    """The knowledge-base file is structurally invalid (e.g. a required category is missing)."""


@dataclass
class Document:
    """One knowledge-base record, normalised and ready to index.

    Attributes:
        id: Stable record id from the source (also seeds the Qdrant point id).
        category: One of `REQUIRED_CATEGORIES`.
        title: Short label shown as the source citation.
        content: The fact itself — what the LLM reads.
        url: Optional link shown with the citation.
        content_hash: Hash of every indexed field, used to detect edits.
    """

    id: str
    category: str
    title: str
    content: str
    url: str | None
    content_hash: str


def fetch_knowledge_records() -> list[dict[str, Any]]:
    """Download the knowledge-base JSON from Supabase Storage.

    Returns:
        The file's flat `records` array — one dict per fact with
        id / category / title / content / url / active.

    Raises:
        SupabaseServiceError: If credentials are missing, the request fails, or the
            file is not JSON with a `records` array.
    """
    if not settings.supabase_url or not settings.supabase_secret_key:
        raise SupabaseServiceError("SUPABASE_URL or SUPABASE_SECRET_KEY is not configured.")

    url = f"{settings.supabase_url}/storage/v1/object/{BUCKET}/{KNOWLEDGE_FILE}"
    headers = {
        "apikey": settings.supabase_secret_key,
        "Authorization": f"Bearer {settings.supabase_secret_key}",
    }
    try:
        response = httpx.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise SupabaseServiceError(f"Supabase Storage request failed: {exc}") from exc
    except ValueError as exc:
        raise SupabaseServiceError(f"Supabase returned invalid JSON: {exc}") from exc

    records = data.get("records")
    if records is None:
        raise SupabaseServiceError(f"'{KNOWLEDGE_FILE}' has no 'records' array.")
    return records


def build_documents(records: list[dict[str, Any]]) -> list[Document]:
    """Validate, normalise and deduplicate raw records.

    Rows with an empty id or content are skipped, inactive rows are dropped, and the
    first occurrence wins for duplicate ids.

    Args:
        records: Raw records from the knowledge-base JSON.

    Returns:
        The documents to index, in source order.

    Raises:
        KnowledgeValidationError: If any required category is absent, so a broken file
            fails loudly instead of silently shrinking the index.
    """
    present_categories = {r.get("category") for r in records}
    missing = [c for c in REQUIRED_CATEGORIES if c not in present_categories]
    if missing:
        raise KnowledgeValidationError(
            f"Knowledge base is missing required categor{'y' if len(missing) == 1 else 'ies'}: {', '.join(missing)}"
        )

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for record in records:
        record_id = record.get("id")
        content = record.get("content")
        if not record_id or not str(record_id).strip() or not content or not str(content).strip():
            continue

        if not _is_active(record.get("active", True)):
            continue

        record_id = _normalize_text(record_id)
        if record_id in seen_ids:
            logger.warning("Duplicate record id '%s' skipped.", record_id)
            continue
        seen_ids.add(record_id)

        content = _normalize_text(content)
        title = _normalize_text(record.get("title") or record_id)
        url = record.get("url")
        url = _normalize_text(url) if url else None
        category = record.get("category", "")

        documents.append(
            Document(
                id=record_id,
                category=category,
                title=title,
                content=content,
                url=url,
                # Every field that reaches the index or the answer, so editing any
                # of them (not just content) triggers a re-sync of that record.
                content_hash=hashlib.sha256(f"{category}␟{title}␟{content}␟{url or ''}".encode()).hexdigest(),
            )
        )
    return documents


def embedding_text(doc: Document) -> str:
    """Build the text that gets embedded for a document.

    Category and title give each record its context; records are atomic facts of at
    most ~220 tokens, so no further chunking is needed.

    Args:
        doc: The document.

    Returns:
        The category and title on the first line, then the content.
    """
    return f"{doc.category.replace('_', ' ')}: {doc.title}\n{doc.content}"


def _normalize_text(value: object) -> str:
    """Trim a value and collapse internal whitespace to single spaces."""
    return re.sub(r"\s+", " ", str(value).strip())


def _is_active(value: Any) -> bool:
    """Interpret an `active` field that may be a bool or a string like "FALSE"."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "inactive", "n"}
