"""Knowledge-base synchronization endpoints: webhook (automation) and admin (manual).

Both run the same idempotent sync; they differ only in the secret that authorises them,
so automation credentials can be rotated without touching the admin token.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import require_admin_token, require_sync_enabled, require_webhook_secret
from app.models.chat import ErrorResponse
from app.models.sync import SyncReport
from app.services.sync_service import run_sync

router = APIRouter()


@router.post(
    "/webhook/sync",
    tags=["webhook"],
    summary="Sync the knowledge base (automation)",
    description=(
        "Re-reads the knowledge base from Supabase and updates the vector index. Send the "
        "`X-Webhook-Secret` header. Any request body is ignored. Idempotent: unchanged records "
        "are skipped, so frequent calls are cheap. Failures are reported as `success: false`."
    ),
    response_model=SyncReport,
    dependencies=[Depends(require_webhook_secret), Depends(require_sync_enabled)],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or wrong X-Webhook-Secret."},
        403: {"model": ErrorResponse, "description": "SYNC_ENABLED is false."},
        503: {"model": ErrorResponse, "description": "WEBHOOK_SECRET is not configured."},
    },
)
async def webhook_sync() -> SyncReport:
    """Synchronize the knowledge base; called by automation with `X-Webhook-Secret`.

    The request body is ignored: sync always re-reads the full knowledge base from
    Supabase, so a forged payload can't inject content. Safe to call repeatedly —
    unchanged records are skipped.

    Returns:
        The sync report (`success: false` with an error if the sync failed).
    """
    return await run_sync()


@router.post(
    "/admin/sync",
    tags=["admin"],
    summary="Sync the knowledge base (manual)",
    description="Same operation as the webhook, authorised with the `x-admin-token` header instead.",
    response_model=SyncReport,
    dependencies=[Depends(require_admin_token), Depends(require_sync_enabled)],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or wrong x-admin-token."},
        403: {"model": ErrorResponse, "description": "SYNC_ENABLED is false."},
        503: {"model": ErrorResponse, "description": "ADMIN_API_TOKEN is not configured."},
    },
)
async def admin_sync() -> SyncReport:
    """Synchronize the knowledge base manually (same operation as the webhook).

    Returns:
        The sync report (`success: false` with an error if the sync failed).
    """
    return await run_sync()
