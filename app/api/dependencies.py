"""FastAPI dependencies (request guards): admin sessions, shared secrets, the sync switch, and rate limits."""

import hashlib
import hmac
import math
import secrets
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Cookie, Header, HTTPException, Request, status

from app.core.config import settings

SESSION_COOKIE = "admin_session"
SESSION_SECONDS = 12 * 3600
LOGIN_ATTEMPTS_PER_MINUTE = 5

# ponytail: per-process signing secret — admins sign in again after a restart (or a
# Render free-tier sleep), and sessions don't span instances. Load it from an env var
# if this ever runs as more than one process.
_PROCESS_SECRET = secrets.token_bytes(32)


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


def check_admin_password(password: str) -> None:
    """Verify the admin password (constant time).

    Raises:
        HTTPException: 503 if `ADMIN_PASSWORD` is unset, 401 if it doesn't match.
    """
    _check_secret(password, settings.admin_password, "Admin password")


def _session_key() -> bytes:
    """Signing key: the process secret mixed with the password, so changing the password ends every session.

    The password itself never appears in a token, and the random process secret means
    a stolen cookie can't be used to brute-force the password offline.
    """
    return hmac.new(_PROCESS_SECRET, settings.admin_password.encode(), hashlib.sha256).digest()


def issue_session_token(now: float | None = None) -> str:
    """Create a signed admin session token: `<expiry>.<hmac>`.

    Args:
        now: Current Unix time (for tests).

    Returns:
        The token to store in the session cookie.
    """
    expires = str(int((now if now is not None else time.time()) + SESSION_SECONDS))
    return f"{expires}.{hmac.new(_session_key(), expires.encode(), hashlib.sha256).hexdigest()}"


def session_is_valid(token: str, now: float | None = None) -> bool:
    """Check a session token's signature and expiry.

    Args:
        token: The cookie value.
        now: Current Unix time (for tests).

    Returns:
        True if the token was issued by this process for the current password and hasn't expired.
    """
    expires, _, signature = token.partition(".")
    if not expires.isdigit() or int(expires) < (now if now is not None else time.time()):
        return False
    expected = hmac.new(_session_key(), expires.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature.encode(), expected.encode())


def require_admin(admin_session: str = Cookie(default="")) -> None:
    """Dependency for admin routes: requires a valid session cookie from `POST /admin/auth`.

    Raises:
        HTTPException: 503 if `ADMIN_PASSWORD` is unset, 401 if the session is missing, forged or expired.
    """
    if not settings.admin_password:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Admin password is not configured on the server.")
    if not session_is_valid(admin_session):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not signed in.")


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


class RateLimiter:
    """Sliding-window limit of requests per client IP per minute, usable as a FastAPI dependency.

    ponytail: in-memory, single-process only — swap for Redis-backed limiting if the
    service is scaled beyond one instance.

    Args:
        limit: Returns the allowed requests per minute (read per request, so settings changes apply).
    """

    WINDOW_SECONDS = 60
    PRUNE_ABOVE = 10_000

    def __init__(self, limit: Callable[[], int]) -> None:
        self._limit = limit
        self._hits: dict[str, list[float]] = defaultdict(list)

    def __call__(self, request: Request) -> None:
        """Count this request against the client's IP.

        Raises:
            HTTPException: 429 with a `Retry-After` header when the limit is exceeded.
        """
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        if len(self._hits) > self.PRUNE_ABOVE:
            for ip in [ip for ip, times in self._hits.items() if not times or now - times[-1] >= self.WINDOW_SECONDS]:
                del self._hits[ip]

        recent = [t for t in self._hits[client_ip] if now - t < self.WINDOW_SECONDS]
        if len(recent) >= self._limit():
            retry_after = max(1, math.ceil(self.WINDOW_SECONDS - (now - recent[0])))
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests.", headers={"Retry-After": str(retry_after)}
            )
        recent.append(now)
        self._hits[client_ip] = recent

    def reset(self) -> None:
        """Forget every client's history (tests)."""
        self._hits.clear()


chat_rate_limit = RateLimiter(lambda: settings.rate_limit_per_minute)
login_rate_limit = RateLimiter(lambda: LOGIN_ATTEMPTS_PER_MINUTE)
