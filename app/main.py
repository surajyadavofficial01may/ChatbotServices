"""FastAPI entry point: middleware, routers, exception handlers, health probes and startup."""

import logging
import re
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_chat, routes_sync
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, request_id_var
from app.services.retrieval_service import preload_models

configure_logging()
logger = logging.getLogger(__name__)

SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm the models in a background thread at startup.

    A Render cold start then doesn't make the first visitor wait for model
    loading, and the port is bound immediately rather than after loading.
    """
    if settings.preload_models:
        threading.Thread(target=preload_models, daemon=True).start()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Retrieval-augmented chatbot over a portfolio knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.include_router(routes_chat.router, prefix=settings.api_prefix)
app.include_router(routes_sync.router, prefix=settings.api_prefix)

register_exception_handlers(app)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Attach a request id to the logging context and the `X-Request-ID` response header.

    A client-supplied id is reused only if it is short and plain (prevents log injection).

    Args:
        request: The incoming request.
        call_next: The next ASGI handler.

    Returns:
        The response, with `X-Request-ID` set.
    """
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", tags=["health"], summary="Service info", description="Service name and status.")
def root() -> dict[str, str]:
    """Identify the service.

    Returns:
        Service name and status.
    """
    return {"name": settings.app_name, "status": "ok"}


@app.get(
    "/health", tags=["health"], summary="Liveness probe", description='`{"status": "healthy"}` while the process is up.'
)
def health_check() -> dict[str, str]:
    """Liveness probe used by Render.

    Returns:
        `{"status": "healthy"}`.
    """
    return {"status": "healthy"}


@app.get(
    f"{settings.api_prefix}/health",
    tags=["health"],
    summary="Configuration health",
    description=(
        "Which dependencies are configured and which retrieval models are active. "
        "Booleans and model names only — never secrets."
    ),
)
def detailed_health() -> dict[str, Any]:
    """Report which dependencies are configured and which retrieval models are active.

    Values are booleans or model names only — no URLs, keys or tokens.

    Returns:
        Status, dependency configuration flags and retrieval model names.
    """
    return {
        "status": "healthy",
        "dependencies": {
            "supabase": bool(settings.supabase_url and settings.supabase_secret_key),
            "qdrant": bool(settings.qdrant_url and settings.qdrant_api_key),
            "llm_provider": settings.llm_provider or None,
            "webhook": bool(settings.webhook_secret),
            "sync_enabled": settings.sync_enabled,
        },
        "retrieval": {
            "embedding_model": settings.embedding_model,
            "sparse_model": settings.sparse_model or None,
            "reranker_model": settings.reranker_model or None,
        },
    }
