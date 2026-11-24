import hashlib
import math
import re

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.api import dependencies, routes_chat
from app.core.config import settings
from app.models.knowledge import KnowledgeEntryIn
from app.services import embedding_service, qdrant_service
from app.services.chunking import build_chunks
from app.services.sync_service import index_chunks

DIM = 32


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No test may reach a real service: a developer's .env is loaded by Settings, so blank it."""
    for name in ("supabase_url", "supabase_service_role_key", "qdrant_url", "qdrant_api_key", "groq_api_key"):
        monkeypatch.setattr(settings, name, "")
    monkeypatch.setattr(settings, "admin_password", "")
    monkeypatch.setattr(settings, "webhook_secret", "")
    monkeypatch.setattr(settings, "no_answer_message", "That information is not available in the knowledge base.")
    qdrant_service.get_client.cache_clear()
    qdrant_service.clear_category_cache()
    dependencies.chat_rate_limit.reset()
    dependencies.login_rate_limit.reset()
    routes_chat._sessions.clear()
    yield
    qdrant_service.clear_category_cache()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bucket(token: str, n: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % n


def fake_dense(text: str) -> list[float]:
    """Deterministic bag-of-words vector: texts sharing words are close."""
    vector = [0.0] * DIM
    for token in _tokens(text):
        vector[_bucket(token, DIM)] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def fake_sparse(text: str) -> qmodels.SparseVector:
    counts: dict[int, float] = {}
    for token in _tokens(text):
        index = _bucket(token, 100_000)
        counts[index] = counts.get(index, 0.0) + 1.0
    return qmodels.SparseVector(indices=list(counts), values=list(counts.values()))


def entry(entry_id: str, category: str, title: str, content: str, **extra) -> dict:
    return {"id": entry_id, "category": category, "title": title, "content": content, **extra}


SAMPLE_ENTRIES = [
    entry("PER001", "Personal_Details", "About Sam", "Sam Lee is a mobile developer who builds a shared calendar app."),
    entry("PER002", "Personal_Details", "Education", "Sam Lee studied informatics at the University of Oslo."),
    entry("PRJ001", "Projects", "Zephyr sync engine", "Zephyr is the engine that syncs calendars offline."),
    entry("PRJ002", "Projects", "Reminders", "Reminders can be sent by email or push notification."),
    entry(
        "ART001",
        "Articles",
        "Pricing calendars",
        "An article on why the Team plan costs 6 dollars per user per month.",
        url="https://example.com/pricing",
    ),
    entry("ART002", "Articles", "Refunds", "An article about refunds within 14 days of the first payment."),
    entry("FAQ001", "FAQs", "Who makes the app?", "The calendar app is made by a small remote team."),
]


def sample_entries() -> list[KnowledgeEntryIn]:
    return [KnowledgeEntryIn.model_validate(e) for e in SAMPLE_ENTRIES]


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(embedding_service, "embed_documents", lambda texts: [fake_dense(t) for t in texts])
    monkeypatch.setattr(embedding_service, "embed_query", fake_dense)
    monkeypatch.setattr(embedding_service, "embed_sparse_documents", lambda texts: [fake_sparse(t) for t in texts])
    monkeypatch.setattr(embedding_service, "embed_sparse_query", fake_sparse)
    monkeypatch.setattr(embedding_service, "embedding_dim", lambda: DIM)
    monkeypatch.setattr(settings, "embedding_model", "fake-dense")
    monkeypatch.setattr(settings, "sparse_model", "fake-bm25")
    monkeypatch.setattr(settings, "reranker_model", "")  # tests opt in to reranking explicitly
    monkeypatch.setattr(settings, "retrieval_score_threshold", 0.0)
    monkeypatch.setattr(settings, "rag_debug", False)


@pytest.fixture
def indexed_client(fake_models):
    client = QdrantClient(":memory:")
    index_chunks(client, build_chunks(sample_entries()))
    return client
