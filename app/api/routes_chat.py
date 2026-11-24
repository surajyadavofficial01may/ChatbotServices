"""Public chat endpoints."""

import threading
import time
import uuid
from collections import OrderedDict

from fastapi import APIRouter, Depends

from app.api.dependencies import chat_rate_limit
from app.core.config import settings
from app.models.chat import ChatAskRequest, ChatAskResponse, ChatRequest, ChatResponse, ErrorResponse
from app.services.retrieval_service import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])

ERRORS = {
    422: {"model": ErrorResponse, "description": "Invalid request body; `details` lists each failing field."},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded; see Retry-After."},
    503: {"model": ErrorResponse, "description": "A dependency is unavailable; see Retry-After."},
}

CHAT_DESCRIPTION = """
Send a message, get a reply grounded **only** in the knowledge base.

- `[n]` in the reply refers to `sources[n-1]`; only cited chunks are returned.
- Questions the knowledge base can't answer get the configured `NO_ANSWER_MESSAGE` and no sources.
- **Follow-ups:** send back the `session_id` from the previous response (the server remembers the
  last few turns), or keep the turns yourself and send `history` (oldest first, roles `user`/`assistant`).
  History only resolves references like "it" or "the first one"; it is never a source of facts.
- Limits: `message` 1–1000 characters; up to 20 history turns of 1–2000 characters.
- Rate limited per client IP; see `Retry-After` on 429 and 503 responses.
"""

# ponytail: in-process session memory (LRU, single instance). Clients that need
# memory across restarts or instances send `history` instead.
MAX_SESSIONS = 1000
_sessions: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()
_sessions_lock = threading.Lock()  # sync routes run in a thread pool


def _recall(session_id: str) -> list[dict[str, str]]:
    """Return a copy of a session's remembered turns (empty for a new session)."""
    with _sessions_lock:
        return list(_sessions.get(session_id, []))


def _remember(session_id: str, message: str, reply: str) -> None:
    """Append one exchange to a session, keeping only the turns the pipeline uses."""
    with _sessions_lock:
        turns = _sessions.pop(session_id, [])
        turns += [{"role": "user", "content": message}, {"role": "assistant", "content": reply[:2000]}]
        _sessions[session_id] = turns[-settings.history_max_turns :]
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


@router.post(
    "",
    summary="Chat",
    description=CHAT_DESCRIPTION,
    response_model=ChatResponse,
    dependencies=[Depends(chat_rate_limit)],
    responses=ERRORS,
)
def chat(payload: ChatRequest) -> ChatResponse:
    """Answer a message from the knowledge base.

    Dependency failures (Qdrant, models, LLM) become 503 responses through the central
    exception handlers in `app.core.exceptions`.

    Args:
        payload: The message, plus an optional session id or explicit history.

    Returns:
        The grounded reply, cited sources, session id and server-side latency.
    """
    started = time.perf_counter()
    session_id = payload.session_id or uuid.uuid4().hex
    if payload.history is not None:
        history = [turn.model_dump() for turn in payload.history]
    else:
        history = _recall(session_id)
    result = answer_question(payload.message, history)
    _remember(session_id, payload.message, result["reply"])
    return ChatResponse(
        reply=result["reply"],
        sources=result["sources"],
        session_id=session_id,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        debug=result.get("debug"),
    )


@router.get("/status", summary="Chat readiness", description='Returns `{"status": "ready"}` when the chat API is up.')
def chat_status() -> dict[str, str]:
    """Report that the chat API is accepting requests.

    Returns:
        `{"status": "ready"}`.
    """
    return {"status": "ready"}


@router.post(
    "/ask",
    summary="Ask a question (deprecated)",
    description="Deprecated: use `POST /chat`. Kept so frontends built on the v0 contract keep working.",
    response_model=ChatAskResponse,
    dependencies=[Depends(chat_rate_limit)],
    responses=ERRORS,
    deprecated=True,
)
def ask(payload: ChatAskRequest) -> dict:
    """Answer a question with the v0 request/response shape.

    Args:
        payload: The question and optional conversation history.

    Returns:
        `{"success", "answer", "sources": [{"category", "title", "url"}], "debug"}`.
    """
    result = answer_question(payload.question, [turn.model_dump() for turn in payload.history])
    return {
        "success": True,
        "answer": result["reply"],
        "sources": [
            {"category": s["category"], "title": s["title"], "url": s["url"] or None} for s in result["sources"]
        ],
        "debug": result.get("debug"),
    }
