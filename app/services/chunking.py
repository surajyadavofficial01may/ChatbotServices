"""Turn knowledge entries into indexable chunks: normalise, split, fingerprint."""

import hashlib
import logging
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from app.core.config import settings
from app.models.knowledge import KnowledgeEntryIn

logger = logging.getLogger(__name__)

SENTENCE_OR_PARAGRAPH = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


@dataclass
class Chunk:
    """One piece of an entry, ready to embed and store as one Qdrant point.

    Attributes:
        entry_id: Id of the Supabase entry this chunk came from.
        chunk_index: Position of the chunk within its entry (0-based).
        chunk_count: How many chunks the entry has, so a partially indexed entry is detectable.
        category: Entry category.
        title: Entry title (shown as the citation; the id when the title is blank).
        text: The chunk's text — what the LLM reads.
        url: Optional link shown with the citation ("" for none).
        entry_hash: Hash of everything that shapes the entry's chunks; any edit changes it.
    """

    entry_id: str
    chunk_index: int
    chunk_count: int
    category: str
    title: str
    text: str
    url: str
    entry_hash: str


def build_chunks(entries: Iterable[KnowledgeEntryIn], chunk_size: int | None = None) -> list[Chunk]:
    """Normalise, deduplicate and chunk the active entries.

    Inactive entries are dropped, and the first occurrence wins for duplicate ids.

    Args:
        entries: Entries from Supabase (or a bulk file).
        chunk_size: Maximum characters per chunk (defaults to `CHUNK_SIZE`).

    Returns:
        Chunks of every active entry, in source order.
    """
    size = chunk_size or settings.chunk_size
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry.active:
            continue
        if entry.id in seen:
            logger.warning("Duplicate entry id '%s' skipped.", entry.id)
            continue
        seen.add(entry.id)

        title = _normalize(entry.title) or entry.id
        pieces = split_text(entry.content, size)
        # Every field that reaches the index or the answer, plus the chunk size, so
        # editing any of them (not just content) re-indexes the entry. sort_order is
        # left out on purpose: reordering rows in the admin grid changes no vector.
        entry_hash = hashlib.sha256(
            "␟".join([entry.category, title, entry.content, entry.url, str(size)]).encode()
        ).hexdigest()
        chunks.extend(
            Chunk(entry.id, i, len(pieces), entry.category, title, piece, entry.url, entry_hash)
            for i, piece in enumerate(pieces)
        )
    return chunks


def split_text(text: str, size: int) -> list[str]:
    """Split text into chunks of at most `size` characters.

    Text that fits stays whole. Otherwise it is cut at paragraph and sentence
    boundaries, falling back to word boundaries for a single overlong sentence, and the
    pieces are packed into chunks of roughly equal length (so no tiny tail chunk is
    left without context).

    ponytail: no overlap between chunks; each chunk is embedded with its entry's title
    and category for context. Add overlap if answers start missing facts that straddle
    a chunk boundary.

    Args:
        text: The entry content.
        size: Maximum characters per chunk.

    Returns:
        One or more non-empty chunks with whitespace collapsed.
    """
    normalized = _normalize(text)
    if len(normalized) <= size:
        return [normalized]
    target = math.ceil(len(normalized) / math.ceil(len(normalized) / size))
    chunks: list[str] = []
    current = ""
    for unit in _units(text, size):
        candidate = f"{current} {unit}" if current else unit
        if current and (len(current) >= target or len(candidate) > size):
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _units(text: str, size: int) -> Iterator[str]:
    """Yield sentences and paragraphs, each cut to at most `size` characters."""
    for piece in SENTENCE_OR_PARAGRAPH.split(text):
        piece = _normalize(piece)
        while len(piece) > size:
            cut = piece.rfind(" ", 0, size + 1)
            cut = cut if cut > 0 else size
            yield piece[:cut]
            piece = piece[cut:].strip()
        if piece:
            yield piece


def embedding_text(chunk: Chunk) -> str:
    """Build the text that gets embedded for a chunk.

    Category and title give every chunk its entry's context.

    Args:
        chunk: The chunk.

    Returns:
        The category and title on the first line, then the text.
    """
    return f"{chunk.category.replace('_', ' ')}: {chunk.title}\n{chunk.text}"


def _normalize(value: str) -> str:
    """Trim a value and collapse internal whitespace to single spaces."""
    return re.sub(r"\s+", " ", value.strip())
