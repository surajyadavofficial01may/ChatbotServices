"""FastAPI entry point: middleware, routers, exception handlers, health probes, admin UI and startup."""

import html
import logging
import re
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_admin, routes_chat, routes_sync
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, request_id_var
from app.services.retrieval_service import preload_models

configure_logging()
logger = logging.getLogger(__name__)

SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm the models in a background thread at startup.

    A cold start then doesn't make the first user wait for model loading, and the
    port is bound immediately rather than after loading.
    """
    if settings.preload_models:
        threading.Thread(target=preload_models, daemon=True).start()
    yield


app = FastAPI(
    title=settings.app_title,
    description=(
        "Retrieval-augmented assistant over a Supabase-managed knowledge base, indexed in Qdrant and "
        "answered by Groq. Integrate with `POST /api/v1/chat`; manage content at `/dataentry`."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.include_router(routes_chat.router, prefix=settings.api_prefix)
app.include_router(routes_sync.router, prefix=settings.api_prefix)
app.include_router(routes_admin.router, prefix=settings.api_prefix)
app.mount("/dataentry/assets", StaticFiles(directory=WEB_DIR), name="dataentry-assets")

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


@app.get("/dataentry", include_in_schema=False)
def dataentry() -> HTMLResponse:
    """Serve the admin studio. The page holds no data; everything it shows comes from the admin API.

    Returns:
        The single-page admin UI with the app title and API prefix filled in, and
        framing forbidden (clickjacking protection).
    """
    page = (WEB_DIR / "dataentry.html").read_text(encoding="utf-8")
    page = page.replace("{{APP_TITLE}}", html.escape(settings.app_title)).replace(
        "{{API_PREFIX}}", html.escape(settings.api_prefix)
    )
    return HTMLResponse(page, headers={"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'"})


@app.get("/", tags=["health"], summary="Service info", description="Service name and status.")
def root() -> dict[str, str]:
    """Identify the service.

    Returns:
        Service name and status.
    """
    return {"name": settings.app_title, "status": "ok"}


@app.get(
    "/health", tags=["health"], summary="Liveness probe", description='`{"status": "healthy"}` while the process is up.'
)
def health_check() -> dict[str, str]:
    """Liveness probe for hosting platforms.

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
            "supabase": bool(settings.supabase_url and settings.supabase_service_role_key),
            "qdrant": bool(settings.qdrant_url and settings.qdrant_api_key),
            "groq": bool(settings.groq_api_key),
            "admin": bool(settings.admin_password),
            "webhook": bool(settings.webhook_secret),
            "sync_enabled": settings.sync_enabled,
        },
        "retrieval": {
            "embedding_model": settings.embedding_model,
            "sparse_model": settings.sparse_model or None,
            "reranker_model": settings.reranker_model or None,
            "llm_model": settings.groq_model,
        },
    }
