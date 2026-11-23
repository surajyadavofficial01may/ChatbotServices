"""Create the Qdrant collection, or recreate it if its schema doesn't match the configured models.

Usage: `python -m scripts.initialize_qdrant`.
"""

from app.services.embedding_service import embedding_dim, sparse_enabled
from app.services.qdrant_service import ensure_collection, get_client

if __name__ == "__main__":
    recreated = ensure_collection(get_client(), embedding_dim(), sparse_enabled())
    print("Collection created." if recreated else "Collection already matches the configured models.")
