"""Public chat endpoints."""

from fastapi import APIRouter, Depends

from app.api.dependencies import enforce_rate_limit
from app.models.chat import ChatAskRequest, ChatAskResponse, ErrorResponse
from app.services.retrieval_service import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])

ASK_DESCRIPTION = """
Answer a question using only the portfolio knowledge base.

- `[n]` in the answer refers to `sources[n-1]`; only cited records are returned.
- Questions the knowledge base can't answer get
  `"That information is not available in the portfolio knowledge base."` with no sources.
- **Follow-ups:** send earlier turns in `history` (oldest first). Each turn's `role` must be
  `"user"` or `"assistant"` — any other role is rejected with 422. History is used only to
  understand references like "it" or "the first one", never as a source of facts.
- Limits: `question` 1–1000 characters; up to 20 history turns of 1–2000 characters.
- Rate limited per client IP; see `Retry-After` on 429 and 503 responses.
"""


@router.get("/status", summary="Chat readiness", description='Returns `{"status": "ready"}` when the chat API is up.')
def chat_status() -> dict[str, str]:
    """Report that the chat API is accepting requests.

    Returns:
        `{"status": "ready"}`.
    """
    return {"status": "ready"}


@router.post(
    "/ask",
    summary="Ask a question",
    description=ASK_DESCRIPTION,
    response_model=ChatAskResponse,
    dependencies=[Depends(enforce_rate_limit)],
    responses={
        422: {"model": ErrorResponse, "description": "Invalid request body; `details` lists each failing field."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded; see Retry-After."},
        503: {"model": ErrorResponse, "description": "A dependency is unavailable; see Retry-After."},
    },
)
def ask(payload: ChatAskRequest) -> dict:
    """Answer a question from the portfolio knowledge base.

    Dependency failures (Qdrant, models, LLM) are turned into 503 responses by the
    central exception handlers in `app.core.exceptions`.

    Args:
        payload: The question and optional conversation history.

    Returns:
        The grounded answer with the sources it cites.
    """
    return answer_question(payload.question, [turn.model_dump() for turn in payload.history])
