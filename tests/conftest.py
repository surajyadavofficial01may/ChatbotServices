import hashlib
import math
import re

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.services import embedding_service
from app.services.data_fetcher import build_documents
from app.services.sync_service import index_documents

DIM = 32


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


SAMPLE_RECORDS = [
    {
        "id": "PER001",
        "category": "Personal_Details",
        "title": "Biography",
        "content": "Alex is a backend engineer based in Pune.",
        "url": None,
        "active": True,
    },
    {
        "id": "PER002",
        "category": "Personal_Details",
        "title": "Education",
        "content": "Alex studied computer science at Pune University with a CGPA of 9.1.",
        "url": None,
        "active": True,
    },
    {
        "id": "EXP001",
        "category": "Work_Experience",
        "title": "Engineer at Acme",
        "content": "Alex works at Acme building payment APIs.",
        "url": None,
        "active": True,
    },
    {
        "id": "SK001",
        "category": "Technical_Skills",
        "title": "Python",
        "content": "Alex writes Python and FastAPI services.",
        "url": None,
        "active": True,
    },
    {
        "id": "PRJ001",
        "category": "Projects",
        "title": "Orbit",
        "content": "Orbit is a voice assistant project for Windows.",
        "url": "https://example.com/orbit",
        "active": True,
    },
    {
        "id": "PRJ002",
        "category": "Projects",
        "title": "Ledger",
        "content": "Ledger is a budgeting project with a React frontend.",
        "url": None,
        "active": True,
    },
    {
        "id": "ART001",
        "category": "Articles",
        "title": "On travel",
        "content": "An article about travel and culture.",
        "url": None,
        "active": True,
    },
    {
        "id": "SOC001",
        "category": "Social_Links",
        "title": "GitHub",
        "content": "Alex's GitHub profile is github.com/alex.",
        "url": "https://github.com/alex",
        "active": True,
    },
    {
        "id": "FAQ001",
        "category": "FAQs",
        "title": "Who is Alex?",
        "content": "Alex is a backend engineer at Acme.",
        "url": None,
        "active": True,
    },
]


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
    index_documents(client, build_documents(SAMPLE_RECORDS))
    return client
