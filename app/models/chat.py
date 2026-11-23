"""Chat request/response schemas and the shared error schema."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """One earlier message in the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatAskRequest(BaseModel):
    """A visitor's question, optionally with earlier turns for follow-up questions.

    History is kept by the client and only used to resolve references such as
    "it" or "the first one"; it is never treated as evidence.
    """

    question: str = Field(min_length=1, max_length=1000, examples=["What projects are in the portfolio?"])
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    """A knowledge-base record cited by an answer."""

    category: str
    title: str
    url: str | None = None


class ChatAskResponse(BaseModel):
    """A grounded answer and the sources it cites."""

    success: bool
    answer: str
    sources: list[Source] = Field(default_factory=list, description="Citation order: [1] in the answer is sources[0].")
    debug: dict[str, Any] | None = Field(default=None, description="Retrieval trace; only when RAG_DEBUG=true.")


class FieldError(BaseModel):
    """One validation failure: which field, and what was wrong with it."""

    field: str = Field(examples=["history.0.role"])
    message: str = Field(examples=["Input should be 'user' or 'assistant'"])


class ErrorResponse(BaseModel):
    """Shape of every error response, from any endpoint."""

    success: bool = False
    error: str
    details: list[FieldError] | None = Field(default=None, description="Present on 422 validation errors only.")
