"""Run the Supabase -> Qdrant sync from the command line, without going through the API.

Usage: `python -m scripts.manual_sync` (prints the JSON sync report).
"""

import asyncio

from app.services.sync_service import knowledge_sync

if __name__ == "__main__":
    report = asyncio.run(knowledge_sync.run())
    print(report.model_dump_json(indent=2))
