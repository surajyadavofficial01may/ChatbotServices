"""FastAPI dependencies (request guards): shared-secret checks, the sync switch, and per-IP rate limiting."""

import hmac
import math
import time
from collections import defaultdict

from fastapi import Header, HTTPException, Request, status

from app.core.config import settings


def _check_secret(provided: str, expected: str, name: str) -> None:
    """Compare a client-supplied secret with the configured one in constant time.

    Args:
        provided: Value sent by the client.
        expected: Value configured on the server.
        name: Human-readable name used in error messages.

    Raises:
        HTTPException: 503 if the server has no secret configured, 401 if it doesn't match.
    """
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"{name} is not configured on the server.")
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid {name.lower()}.")


def require_admin_token(x_admin_token: str = Header(default="")) -> None:
    """Dependency for admin routes: requires the `x-admin-token` header.

    Args:
        x_admin_token: Value of the `x-admin-token` request header.

    Raises:
        HTTPException: 503 if `ADMIN_API_TOKEN` is unset, 401 if the header is wrong.
    """
    _check_secret(x_admin_token, settings.admin_api_token, "Admin token")


def require_webhook_secret(x_webhook_secret: str = Header(default="")) -> None:
    """Dependency for webhook routes: requires the `X-Webhook-Secret` header.

    Args:
        x_webhook_secret: Value of the `X-Webhook-Secret` request header.

    Raises:
        HTTPException: 503 if `WEBHOOK_SECRET` is unset, 401 if the header is wrong.
    """
    _check_secret(x_webhook_secret, settings.webhook_secret, "Webhook secret")


def require_sync_enabled() -> None:
    """Dependency for sync routes: rejects calls while `SYNC_ENABLED=false`.

    Raises:
        HTTPException: 403 when synchronization is disabled.
    """
    if not settings.sync_enabled:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Synchronization is disabled on this server.")


# ponytail: in-memory sliding-window limiter, single-process only — swap for
# Redis-backed limiting if Render is scaled beyond one instance.
_WINDOW_SECONDS = 60
_PRUNE_ABOVE = 10_000
_hits: dict[str, list[float]] = defaultdict(list)


def enforce_rate_limit(request: Request) -> None:
    """Dependency limiting each client IP to `RATE_LIMIT_PER_MINUTE` requests per minute.

    Args:
        request: The incoming request, used for the client address.

    Raises:
        HTTPException: 429 with a `Retry-After` header when the limit is exceeded.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    if len(_hits) > _PRUNE_ABOVE:
        for ip in [ip for ip, times in _hits.items() if not times or now - times[-1] >= _WINDOW_SECONDS]:
            del _hits[ip]

    recent = [t for t in _hits[client_ip] if now - t < _WINDOW_SECONDS]
    if len(recent) >= settings.rate_limit_per_minute:
        retry_after = max(1, math.ceil(_WINDOW_SECONDS - (now - recent[0])))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests.", headers={"Retry-After": str(retry_after)}
        )
    recent.append(now)
    _hits[client_ip] = recent
