"""Centralised error types and FastAPI exception handlers.

Every error response has the same shape: `{"success": false, "error": "<message>"}`.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from starlette.exceptions import HTTPException  # base class: also covers routing 404/405

logger = logging.getLogger(__name__)

DEFAULT_RETRY_AFTER_SECONDS = 30


class ExternalServiceError(Exception):
    """A dependency (Qdrant, Supabase, the LLM provider, a local model) is unavailable.

    Mapped to HTTP 503 with a `Retry-After` header: the request was fine, the
    backend can't serve it right now.

    Attributes:
        retry_after: Seconds the client should wait before retrying.
        expose: Subclasses whose messages are written to be safe and actionable (no
            URLs, keys or payloads) set this so the client sees the message itself.
    """

    expose = False

    def __init__(self, message: str, retry_after: int = DEFAULT_RETRY_AFTER_SECONDS) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _error(status_code: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    """Build an error response in the API's standard shape.

    Args:
        status_code: HTTP status code.
        message: Client-safe error message.
        headers: Optional extra response headers.

    Returns:
        A JSON response `{"success": false, "error": message}`.
    """
    return JSONResponse(status_code=status_code, content={"success": False, "error": message}, headers=headers)


async def _external_service_error(request: Request, exc: ExternalServiceError) -> JSONResponse:
    """Translate a dependency failure into 503 + Retry-After (details stay in the log)."""
    logger.error("Dependency unavailable on %s: %s: %s", request.url.path, type(exc).__name__, exc)
    message = str(exc) if exc.expose else "The service is temporarily unavailable. Please try again shortly."
    return _error(503, message, {"Retry-After": str(exc.retry_after)})


async def _qdrant_client_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate a raw qdrant-client failure into the same 503 as other dependencies."""
    return await _external_service_error(request, ExternalServiceError(str(exc)))


async def _http_exception(_: Request, exc: HTTPException) -> JSONResponse:
    """Render FastAPI/Starlette HTTP errors (401, 403, 429...) in the standard shape."""
    return _error(exc.status_code, str(exc.detail), getattr(exc, "headers", None))


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Reject malformed requests, saying which field failed and why.

    Only the field path and Pydantic's message are returned — never the submitted
    value, so client input is not echoed back.
    """
    details = [_field_error(err) for err in exc.errors()]
    return JSONResponse(status_code=422, content={"success": False, "error": "Invalid request.", "details": details})


def _field_error(err: dict[str, Any]) -> dict[str, str]:
    """Turn one Pydantic error into `{"field", "message"}` (dotted path, e.g. `history.0.role`)."""
    if err["type"] == "json_invalid":  # loc is ("body", <char offset>) — not a field
        return {"field": "body", "message": f"Malformed JSON: {err.get('ctx', {}).get('error', 'decode error')}"}
    field = ".".join(str(part) for part in err["loc"] if part != "body")
    return {"field": field or "body", "message": err["msg"]}


async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: log the stack trace server-side, never return it to the client."""
    logger.exception("Unhandled error on %s", request.url.path)
    return _error(500, "Internal server error.")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the application.

    Args:
        app: The FastAPI application.
    """
    app.add_exception_handler(ExternalServiceError, _external_service_error)
    app.add_exception_handler(UnexpectedResponse, _qdrant_client_error)
    app.add_exception_handler(ResponseHandlingException, _qdrant_client_error)
    app.add_exception_handler(HTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)
