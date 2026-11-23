"""Structured logging with a per-request correlation id on every line."""

import logging
import sys
from contextvars import ContextVar

from app.core.config import settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    """Stamps each log record with the current request id (or "-" outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach `request_id` to the record.

        Args:
            record: The log record being emitted.

        Returns:
            Always True; the filter only enriches records.
        """
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    """Configure root logging: one stdout handler, request id in every line.

    INFO in production, DEBUG elsewhere. Chatty third-party loggers are capped at
    WARNING — httpx in particular would otherwise log full request URLs.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    logging.basicConfig(
        level=logging.INFO if settings.app_env == "production" else logging.DEBUG, handlers=[handler], force=True
    )
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
