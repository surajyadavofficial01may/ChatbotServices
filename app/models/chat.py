"""Chat request/response schemas and the shared error schema."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SESSION_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"


class ChatTurn(BaseModel):
    """One earlier message in the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    """A user message.

    Follow-up questions ("what does *it* cost?") need the earlier turns. Either send
    back the `session_id` from the previous response and the server remembers them,
    or keep them yourself and send `history` (stateless clients). History is only used
    to resolve references, never as a source of facts.
    """

    message: str = Field(min_length=1, max_length=1000, examples=["How do I reset my password?"])
    session_id: str | None = Field(
        default=None, pattern=SESSION_ID_PATTERN, description="From a previous response; omit to start a new session."
    )
    history: list[ChatTurn] | None = Field(
        default=None, max_length=20, description="Earlier turns, oldest first. Overrides the session's memory."
    )


class Source(BaseModel):
    """A knowledge-base chunk cited by the reply."""

    entry_id: str
    title: str
    category: str
    url: str = Field(default="", description='Source link ("" when the row has none).')
    snippet: str = Field(description="The cited chunk's text (truncated).")
    score: float | None = Field(default=None, description="Relevance score that ranked this chunk.")


class ChatResponse(BaseModel):
    """A grounded reply and the sources it cites."""

    reply: str
    sources: list[Source] = Field(default_factory=list, description="Citation order: [1] in the reply is sources[0].")
    session_id: str = Field(description="Send this back with the next message to continue the conversation.")
    latency_ms: float
    debug: dict[str, Any] | None = Field(default=None, description="Retrieval trace; only when RAG_DEBUG=true.")


# --- Deprecated v0 contract (POST /api/v1/chat/ask), kept so existing frontends keep working ---


class ChatAskRequest(BaseModel):
    """Deprecated: use `ChatRequest`."""

    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class LegacySource(BaseModel):
    """Deprecated: use `Source`."""

    category: str
    title: str
    url: str | None = None


class ChatAskResponse(BaseModel):
    """Deprecated: use `ChatResponse`."""

    success: bool
    answer: str
    sources: list[LegacySource] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


class FieldError(BaseModel):
    """One validation failure: which field, and what was wrong with it."""

    field: str = Field(examples=["history.0.role"])
    message: str = Field(examples=["Input should be 'user' or 'assistant'"])


class ErrorResponse(BaseModel):
    """Shape of every error response, from any endpoint."""

    success: bool = False
    error: str
    details: list[FieldError] | None = Field(default=None, description="Present on 422 validation errors only.")
