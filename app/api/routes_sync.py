"""Knowledge-base synchronization webhook, for automation (GitHub Actions, Supabase webhooks, cron).

The admin UI runs the same sync through `POST /admin/sync`; the two differ only in what
authorises them, so automation credentials can be rotated without touching the admin password.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import require_sync_enabled, require_webhook_secret
from app.models.chat import ErrorResponse
from app.models.sync import SyncReport
from app.services import sync_service

router = APIRouter()


@router.post(
    "/webhook/sync",
    tags=["webhook"],
    summary="Sync the knowledge base (automation)",
    description=(
        "Re-reads every active entry from Supabase and updates the Qdrant index. Send the "
        "`X-Webhook-Secret` header. Any request body is ignored. Idempotent: unchanged entries "
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

    The request body is ignored: sync always re-reads Supabase, so a forged payload
    can't inject content.

    Returns:
        The sync report (`success: false` with an error if the sync failed).
    """
    return await sync_service.knowledge_sync.run()
