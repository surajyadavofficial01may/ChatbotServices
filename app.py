"""Local development entry point: `python app.py` (Render runs uvicorn directly)."""

import os

import uvicorn

from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=settings.app_env != "production",
        # Only app code: editing tests, docs, data or benchmark output must not restart the server.
        reload_dirs=["app"],
    )
