"""Service-layer tests, one section per service. Fully offline: fake embedder, in-memory Qdrant, stubbed HTTP."""

import asyncio
import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.core.config import settings
from app.models.knowledge import CATEGORIES, KnowledgeEntryIn
from app.services import (
    chunking,
    embedding_service,
    llm_service,
    qdrant_service,
    rerank_service,
    retrieval_service,
    supabase_repository,
)
from app.services.chunking import Chunk, build_chunks, split_text
from app.services.embedding_service import EmbeddingServiceError
from app.services.llm_service import LLMServiceError, answer_system_prompt
from app.services.query_service import QueryPlan, needs_rewrite, plan_query, route_categories
from app.services.retrieval_service import Candidate, apply_citations, build_context, fuse, retrieve, select
from app.services.supabase_repository import DuplicateEntryError, SupabaseRepository, SupabaseServiceError
from app.services.sync_service import index_chunks
from tests.conftest import SAMPLE_ENTRIES, sample_entries

ROOT = Path(__file__).resolve().parent.parent
NO_ANSWER = "That information is not available in the knowledge base."

# ----------------------------------------------------------------------------
# models: KnowledgeEntry schema
# ----------------------------------------------------------------------------


def _entry(entry_id="E", category="Projects", **fields):
    return KnowledgeEntryIn.model_validate({"id": entry_id, "category": category, "content": "C", **fields})


def test_entry_defaults_match_the_table_defaults():
    entry = _entry(url=None)
    assert (entry.title, entry.url, entry.active, entry.sort_order) == ("", "", True, 0)


def test_rows_from_the_tab_views_and_spreadsheets_are_accepted():
    entry = KnowledgeEntryIn.model_validate(
        {
            "id": "PRJ001",
            "category": "Projects",
            "title": "T",
            "content": "C",
            "url": "https://x.dev",
            "active": "FALSE",
        }
    )
    assert entry.url == "https://x.dev" and entry.active is False


@pytest.mark.parametrize(
    "field, value",
    [
        ("id", "has space"),
        ("id", ""),
        ("category", "General"),  # only the seven tabs
        ("url", "javascript:alert(1)"),
        ("content", ""),
        ("title", "x" * 201),
        ("sort_order", -1),
    ],
)
def test_invalid_entries_are_rejected(field, value):
    with pytest.raises(ValueError):
        _entry(**{field: value})


def test_id_and_category_are_required():
    with pytest.raises(ValueError):
        KnowledgeEntryIn.model_validate({"content": "C"})


def test_schema_drives_the_admin_grid():
    schema = KnowledgeEntryIn.model_json_schema()
    assert list(schema["properties"]) == ["id", "category", "title", "content", "url", "active", "sort_order"]
    assert schema["required"] == ["id", "category", "content"]
    category = schema["properties"]["category"]
    assert category["enum"] == list(CATEGORIES) and category["ui"] == "tabs"
    assert category["id_prefixes"]["Projects"] == "PRJ" and set(category["id_prefixes"]) == set(CATEGORIES)
    assert schema["properties"]["content"]["ui"] == "textarea"


def test_categories_match_the_sql_check_constraint():
    sql = (ROOT / "supabase" / "schema.sql").read_text(encoding="utf-8")
    check = sql[sql.index("category IN (") : sql.index(")", sql.index("category IN ("))]
    assert re.findall(r"'([A-Za-z_]+)'", check) == list(CATEGORIES)
    for category in CATEGORIES:
        assert f'VIEW public."{category}" WITH (security_invoker = true)' in sql


def test_shipped_sample_dataset_is_valid():
    rows = json.loads((ROOT / "data" / "sample_entries.json").read_text(encoding="utf-8"))
    entries = [KnowledgeEntryIn.model_validate(row) for row in rows]
    assert len({e.id for e in entries}) == len(entries)
    assert {e.category for e in entries} == set(CATEGORIES)  # every tab has an example
    assert any(not e.active for e in entries)  # demonstrates exclusion
    assert any(c.chunk_count > 1 for c in build_chunks(entries))  # demonstrates chunking


# ----------------------------------------------------------------------------
# chunking
# ----------------------------------------------------------------------------


def test_short_content_is_one_chunk_with_whitespace_collapsed():
    assert split_text("  a \n\n b  ", 100) == ["a b"]


def test_long_content_splits_on_sentences_within_the_limit_and_loses_nothing():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, 300)
    assert len(chunks) > 1 and all(len(c) <= 300 for c in chunks)
    assert " ".join(chunks).split() == text.split()
    assert all(c.endswith(".") for c in chunks)  # cut between sentences, not mid-sentence


def test_chunks_are_evenly_sized_not_a_tiny_tail():
    chunks = split_text("A" * 50 + ". " + "B" * 50 + ". " + "C" * 50 + ".", 110)
    assert len(chunks) == 2 and min(len(c) for c in chunks) > 40


def test_an_unbreakable_word_is_hard_cut():
    assert [len(c) for c in split_text("x" * 250, 100)] == [100, 100, 50]


def test_inactive_and_duplicate_entries_are_dropped():
    entries = sample_entries() + [_entry("PER001", title="dup"), _entry("OFF", active=False)]
    ids = {c.entry_id for c in build_chunks(entries)}
    assert "OFF" not in ids and len(ids) == len(SAMPLE_ENTRIES)


def test_blank_title_falls_back_to_the_id():
    assert build_chunks([_entry("PRJ009")])[0].title == "PRJ009"


@pytest.mark.parametrize(
    "field, value",
    [("title", "New"), ("category", "Articles"), ("content", "Other"), ("url", "https://n.dev")],
)
def test_any_indexed_field_changes_the_entry_hash(field, value):
    base = _entry(title="T")
    changed = base.model_copy(update={field: value})
    assert build_chunks([base])[0].entry_hash != build_chunks([changed])[0].entry_hash


def test_reordering_rows_never_re_embeds():
    base = _entry(title="T", sort_order=1)
    assert build_chunks([base])[0].entry_hash == build_chunks([base.model_copy(update={"sort_order": 9})])[0].entry_hash


def test_chunk_size_is_part_of_the_hash():
    entry = _entry(title="T")
    assert build_chunks([entry], 100)[0].entry_hash != build_chunks([entry], 200)[0].entry_hash


def test_embedding_text_carries_entry_context():
    chunk = build_chunks([_entry(category="Work_Experience", title="Engineer at Acme", content="Built APIs.")])[0]
    assert chunking.embedding_text(chunk) == "Work Experience: Engineer at Acme\nBuilt APIs."


# ----------------------------------------------------------------------------
# supabase_repository (httpx MockTransport)
# ----------------------------------------------------------------------------


def _configure_supabase(monkeypatch):
    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co/")
    monkeypatch.setattr(settings, "supabase_service_role_key", "sb_secret_test")


def _repo(handler):
    return SupabaseRepository(transport=httpx.MockTransport(handler))


def _row(entry_id, **extra):
    return {
        "id": entry_id,
        "category": "Projects",
        "title": "T",
        "content": "C",
        "updated_at": "2026-01-01T00:00:00Z",
        **extra,
    }


def test_repository_requires_configuration():
    with pytest.raises(SupabaseServiceError, match="SUPABASE_URL"):
        asyncio.run(_repo(lambda r: httpx.Response(200, json=[])).list_entries())


def test_list_entries_pages_past_the_row_cap(monkeypatch):
    _configure_supabase(monkeypatch)
    monkeypatch.setattr(supabase_repository, "PAGE_SIZE", 2)
    rows = [_row(f"E{i}") for i in range(5)]
    seen = []

    def handler(request):
        seen.append(request)
        offset, limit = int(request.url.params["offset"]), int(request.url.params["limit"])
        return httpx.Response(200, json=rows[offset : offset + limit])

    entries = asyncio.run(_repo(handler).list_entries(active_only=True))
    assert [e.id for e in entries] == ["E0", "E1", "E2", "E3", "E4"] and len(seen) == 3
    assert seen[0].url.path == "/rest/v1/portfolio_knowledge_base"
    assert seen[0].url.params["active"] == "eq.true"
    assert seen[0].headers["apikey"] == "sb_secret_test"


def test_create_sends_writable_columns_and_maps_conflict(monkeypatch):
    _configure_supabase(monkeypatch)
    sent = {}

    def handler(request):
        sent["body"] = json.loads(request.content)
        sent["prefer"] = request.headers["prefer"]
        return httpx.Response(201, json=[_row("E1")])

    created = asyncio.run(_repo(handler).create_entry(_entry("E1", title="T")))
    assert created.id == "E1" and sent["prefer"] == "return=representation"
    assert set(sent["body"]) == {"id", "category", "title", "content", "url", "active", "sort_order"}

    with pytest.raises(DuplicateEntryError):
        asyncio.run(_repo(lambda r: httpx.Response(409, json={"code": "23505"})).create_entry(created))


def test_update_never_sends_the_id_and_reports_missing_rows(monkeypatch):
    _configure_supabase(monkeypatch)
    sent = {}

    def handler(request):
        sent["filter"] = request.url.params["id"]
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    result = asyncio.run(_repo(handler).update_entry("E1", _entry("OTHER", title="T")))
    assert result is None and sent["filter"] == "eq.E1" and "id" not in sent["body"]


def test_bulk_upsert_merges_duplicates(monkeypatch):
    _configure_supabase(monkeypatch)
    sent = {}

    def handler(request):
        sent["prefer"] = request.headers["prefer"]
        return httpx.Response(201, json=json.loads(request.content))

    count = asyncio.run(_repo(handler).upsert_entries(sample_entries()))
    assert count == len(SAMPLE_ENTRIES) and "resolution=merge-duplicates" in sent["prefer"]


def test_missing_table_error_explains_the_fix(monkeypatch):
    _configure_supabase(monkeypatch)
    reply = httpx.Response(404, json={"code": "PGRST205", "message": "Could not find the table"})
    with pytest.raises(SupabaseServiceError, match="supabase/schema.sql"):
        asyncio.run(_repo(lambda r: reply).list_entries())


def test_network_errors_are_wrapped_without_details(monkeypatch):
    _configure_supabase(monkeypatch)

    def boom(request):
        raise httpx.ConnectError("connection refused to https://example.supabase.co?apikey=secret")

    with pytest.raises(SupabaseServiceError) as err:
        asyncio.run(_repo(boom).list_entries())
    assert "secret" not in str(err.value) and "ConnectError" in str(err.value)


# ----------------------------------------------------------------------------
# qdrant_service
# ----------------------------------------------------------------------------

DIM = 4
VERSION = "test-version"


def _chunk(entry_id: str, category: str = "Features", index: int = 0, count: int = 1) -> Chunk:
    return Chunk(entry_id, index, count, category, entry_id, f"content {entry_id}", "", f"hash-{entry_id}")


def _client() -> QdrantClient:
    client = QdrantClient(":memory:")
    qdrant_service.ensure_collection(client, DIM, with_sparse=False)
    return client


def _vec(i: int) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(DIM)]


def test_upsert_then_query_finds_matching_vector():
    client = _client()
    qdrant_service.upsert_chunks(client, [_chunk("feat-1")], [_vec(0)], None, VERSION)
    results = qdrant_service.query_candidates(client, _vec(0), None, 1, VERSION, [])
    assert results["dense"][0].payload["entry_id"] == "feat-1"


def test_repeated_upsert_is_idempotent():
    client = _client()
    for _ in range(2):
        qdrant_service.upsert_chunks(client, [_chunk("feat-1")], [_vec(0)], None, VERSION)
    assert client.count(settings.qdrant_collection_name).count == 1


def test_delete_removes_every_chunk_of_an_entry():
    client = _client()
    chunks = [_chunk("long", index=i, count=3) for i in range(3)] + [_chunk("other")]
    qdrant_service.upsert_chunks(client, chunks, [_vec(i % DIM) for i in range(4)], None, VERSION)
    qdrant_service.delete_entries(client, ["long"])
    assert set(qdrant_service.scan_index(client)[0]) == {"other"}


def test_partially_indexed_entry_gets_an_empty_fingerprint():
    client = _client()
    qdrant_service.upsert_chunks(client, [_chunk("long", index=0, count=3)], [_vec(0)], None, VERSION)
    assert qdrant_service.scan_index(client)[0] == {"long": ""}


def test_category_retriever_only_returns_that_category():
    client = _client()
    qdrant_service.upsert_chunks(client, [_chunk("feat-1"), _chunk("faq-1", "FAQs")], [_vec(0), _vec(0)], None, VERSION)
    results = qdrant_service.query_candidates(client, _vec(0), None, 5, VERSION, ["FAQs"])
    assert [p.payload["entry_id"] for p in results["category:FAQs"]] == ["faq-1"]


def test_categories_come_from_the_index_and_are_cached():
    client = _client()
    qdrant_service.upsert_chunks(
        client, [_chunk("a", "Billing"), _chunk("b", "FAQs")], [_vec(0), _vec(1)], None, VERSION
    )
    assert sorted(qdrant_service.list_categories(client, VERSION)) == ["Billing", "FAQs"]
    qdrant_service.upsert_chunks(client, [_chunk("c", "New")], [_vec(2)], None, VERSION)
    assert "New" not in qdrant_service.list_categories(client, VERSION)  # cached
    qdrant_service.clear_category_cache()
    assert "New" in qdrant_service.list_categories(client, VERSION)


def test_category_lookup_failure_never_fails_a_request():
    broken = MagicMock()
    broken.facet.side_effect = RuntimeError("down")
    assert qdrant_service.list_categories(broken, VERSION) == []


def test_matching_schema_is_left_alone():
    client = _client()
    assert qdrant_service.ensure_collection(client, DIM, with_sparse=False) is False


def _existing_collection(payload_schema):
    vectors = {qdrant_service.DENSE: SimpleNamespace(size=DIM)}
    info = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=vectors, sparse_vectors=None)),
        payload_schema=payload_schema,
    )
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = info
    return client


def test_missing_payload_index_is_repaired_on_an_existing_collection():
    # Strict-mode Qdrant Cloud rejects filters on unindexed fields: a right-shaped
    # collection without the embedding_version index must still be fixed by sync.
    client = _existing_collection({"category": object()})
    assert qdrant_service.ensure_collection(client, DIM, with_sparse=False) is False
    created = [call.kwargs["field_name"] for call in client.create_payload_index.call_args_list]
    assert created == ["embedding_version", "entry_id"]
    client.delete_collection.assert_not_called()


def test_points_from_before_chunking_are_removed_by_the_next_sync(fake_models):
    from qdrant_client.http import models as qmodels

    client = QdrantClient(":memory:")
    _index(client, SAMPLE_ENTRIES)
    legacy = qmodels.PointStruct(
        id=str(uuid.uuid4()),
        vector={qdrant_service.DENSE: [1.0] * 32},
        payload={"record_id": "OLD001", "content": "old one-point-per-record layout"},
    )
    client.upsert(settings.qdrant_collection_name, points=[legacy])
    assert qdrant_service.scan_index(client)[1] == [legacy.id]

    report = _index(client, SAMPLE_ENTRIES)
    assert report["unchanged"] == len(SAMPLE_ENTRIES)
    assert client.count(settings.qdrant_collection_name).count == len(SAMPLE_ENTRIES)


def test_complete_collection_gets_no_index_calls():
    client = _existing_collection({field: object() for field in qdrant_service.FILTERED_FIELDS})
    qdrant_service.ensure_collection(client, DIM, with_sparse=False)
    client.create_payload_index.assert_not_called()


# ----------------------------------------------------------------------------
# embedding_service
# ----------------------------------------------------------------------------


class _FakeDenseModel:
    def __init__(self, dim=384):
        self.dim, self.seen = dim, []

    def embed(self, texts):
        self.seen.extend(texts)
        return (np.ones(self.dim, dtype=np.float32) for _ in texts)

    def query_embed(self, text):
        self.seen.append(text)
        yield np.ones(self.dim, dtype=np.float32)


@pytest.fixture
def fake_dense_model(monkeypatch):
    model = _FakeDenseModel()
    monkeypatch.setattr(embedding_service, "_dense", lambda: model)
    monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(settings, "embedding_query_prefix", None)
    embedding_service._cached_query.cache_clear()
    yield model
    embedding_service._cached_query.cache_clear()


def test_empty_input_returns_empty_list():
    assert embedding_service.embed_documents([]) == []


def test_bge_query_gets_retrieval_instruction_but_documents_do_not(fake_dense_model):
    embedding_service.embed_query("What is Zephyr?")
    embedding_service.embed_documents(["Zephyr is a sync engine."])
    assert fake_dense_model.seen[0].startswith("Represent this sentence for searching relevant passages: ")
    assert fake_dense_model.seen[1] == "Zephyr is a sync engine."


def test_prefix_override_replaces_model_default(fake_dense_model, monkeypatch):
    monkeypatch.setattr(settings, "embedding_query_prefix", "")
    embedding_service.embed_query("plain")
    assert fake_dense_model.seen == ["plain"]


def test_repeated_queries_hit_the_cache(fake_dense_model):
    embedding_service.embed_query("same question")
    embedding_service.embed_query("same question")
    assert len(fake_dense_model.seen) == 1


def test_wrong_vector_dimension_is_rejected(fake_dense_model):
    fake_dense_model.dim = 100  # bge-small must produce 384
    with pytest.raises(EmbeddingServiceError, match="dimension mismatch"):
        embedding_service.embed_documents(["x"])


def test_embedding_version_changes_with_any_vector_shaping_setting(monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(settings, "sparse_model", "Qdrant/bm25")
    monkeypatch.setattr(settings, "embedding_query_prefix", None)
    base = embedding_service.embedding_version()

    monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-base-en-v1.5")
    assert embedding_service.embedding_version() != base
    monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(settings, "sparse_model", "")
    assert embedding_service.embedding_version() != base
    monkeypatch.setattr(settings, "sparse_model", "Qdrant/bm25")
    monkeypatch.setattr(embedding_service, "TEXT_TEMPLATE_VERSION", "999")
    assert embedding_service.embedding_version() != base


def test_known_model_dimension_comes_from_model_registry(monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-base-en-v1.5")
    assert embedding_service.embedding_dim() == 768


def test_concurrent_first_requests_load_a_model_only_once():
    loads = []

    def slow_factory():
        loads.append(1)
        time.sleep(0.05)
        return object()

    threads = [
        threading.Thread(target=embedding_service.load_model, args=("test", "once", slow_factory)) for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loads) == 1


def test_model_load_failure_raises_service_error():
    def broken():
        raise RuntimeError("download failed")

    with pytest.raises(EmbeddingServiceError, match="Failed to load"):
        embedding_service.load_model("test", "broken-model", broken)


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1", reason="Downloads the real FastEmbed models; set RUN_MODEL_TESTS=1 to enable."
)
def test_real_models_produce_expected_shapes(monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(settings, "sparse_model", "Qdrant/bm25")
    assert len(embedding_service.embed_documents(["Acme Notes is a note-taking app."])[0]) == 384
    assert len(embedding_service.embed_query("What does Acme Notes do?")) == 384
    sparse = embedding_service.embed_sparse_query("two-factor authentication")
    assert len(sparse.indices) == len(sparse.values) > 0


# ----------------------------------------------------------------------------
# query_service
# ----------------------------------------------------------------------------

ROUTING_CATEGORIES = ["Projects", "Technical_Skills", "Articles", "Billing", "Security & Privacy", "Policy", "FAQs"]
HISTORY = [
    {"role": "user", "content": "What is Zephyr?"},
    {"role": "assistant", "content": "Zephyr is the sync engine."},
]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What projects has Alex worked on?", ["Projects"]),  # plural category = list question
        ("List all the articles you have.", ["Articles"]),
        ("What are your technical skills?", ["Technical_Skills"]),
        ("Which skills do you have?", ["Technical_Skills"]),  # head noun alone
        ("List every billing topic", ["Billing"]),  # explicit cue, singular name
        ("Show me all your policies", ["Policy"]),  # y -> ies plural
        ("What is the security and privacy policy?", []),  # names categories, but not a list question
        ("What technologies did Alex use to build Zephyr?", []),
        ("Where did Alex study?", []),
    ],
)
def test_category_routing_only_for_list_questions(question, expected):
    assert route_categories(question, ROUTING_CATEGORIES)[0] == expected


def test_routing_uses_the_categories_passed_in():
    assert plan_query("What projects exist?", categories=ROUTING_CATEGORIES).list_mode
    assert not plan_query("What projects exist?", categories=[]).list_mode  # unknown category: no routing


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What latency does it achieve?", True),
        ("What tech stack was used for the first one?", True),
        ("Which neural networks were used there?", True),
        ("And Aiden?", True),  # very short follow-up
        ("What is his email address?", True),  # pronouns refer to earlier turns
        ("Which databases does Nimbus support?", False),
    ],
)
def test_follow_up_detection(question, expected):
    assert needs_rewrite(question) is expected


def test_no_history_means_no_llm_call(monkeypatch):
    monkeypatch.setattr(llm_service, "complete", lambda *a: pytest.fail("LLM must not be called"))
    plan = plan_query("What latency does it achieve?")
    assert plan.retrieval_query == "What latency does it achieve?" and not plan.rewritten


def test_follow_up_is_rewritten_into_standalone_query(monkeypatch):
    monkeypatch.setattr(
        llm_service, "complete", lambda system, user: "Standalone query: What latency does Zephyr achieve?"
    )
    plan = plan_query("What latency does it achieve?", HISTORY)
    assert plan.rewritten and plan.retrieval_query == "What latency does Zephyr achieve?"


def test_standalone_question_with_history_is_not_rewritten(monkeypatch):
    monkeypatch.setattr(llm_service, "complete", lambda *a: pytest.fail("LLM must not be called"))
    assert not plan_query("Which databases does Nimbus support?", HISTORY).rewritten


def test_rewrite_failure_falls_back_to_original_question(monkeypatch):
    def fail(*_):
        raise llm_service.LLMServiceError("rate limited")

    monkeypatch.setattr(llm_service, "complete", fail)
    plan = plan_query("What latency does it achieve?", HISTORY)
    assert plan.retrieval_query == "What latency does it achieve?" and not plan.rewritten


def test_runaway_rewrite_output_is_discarded(monkeypatch):
    monkeypatch.setattr(llm_service, "complete", lambda *a: "x" * 500)
    assert plan_query("What latency does it achieve?", HISTORY).retrieval_query == "What latency does it achieve?"


def test_only_recent_history_reaches_the_rewriter(monkeypatch):
    monkeypatch.setattr(settings, "history_max_turns", 2)
    seen = {}

    def capture(system, user):
        seen["user"] = user
        return "rewritten"

    monkeypatch.setattr(llm_service, "complete", capture)
    old = [{"role": "user", "content": "OLD-TOPIC"}, {"role": "assistant", "content": "old answer"}]
    plan_query("tell me more about it", old + HISTORY)
    assert "OLD-TOPIC" not in seen["user"] and "What is Zephyr?" in seen["user"]


# ----------------------------------------------------------------------------
# retrieval_service: fusion, gate, context, retrieval
# ----------------------------------------------------------------------------


def _point(entry_id, score, category="Features", chunk_index=0):
    return SimpleNamespace(
        score=score,
        payload={
            "entry_id": entry_id,
            "chunk_index": chunk_index,
            "category": category,
            "title": entry_id,
            "content": f"content of {entry_id}",
            "url": "",
        },
    )


def _plan(query, categories=None, list_mode=False):
    return QueryPlan(question=query, retrieval_query=query, categories=categories or [], list_mode=list_mode)


def test_rrf_rewards_documents_ranked_well_by_both_retrievers():
    results = {
        "dense": [_point("A", 0.9), _point("B", 0.8), _point("C", 0.7)],
        "sparse": [_point("C", 12.0), _point("A", 9.0)],
    }
    fused = fuse(results, k=60)
    assert [c.entry_id for c in fused] == ["A", "C", "B"]
    assert fused[0].ranks == {"dense": 1, "sparse": 2}


def test_points_from_before_chunking_still_answer_until_the_next_sync():
    legacy = SimpleNamespace(
        score=0.7,
        payload={"record_id": "PRJ001", "category": "Projects", "title": "T", "content": "C", "url": "https://x"},
    )
    [candidate] = fuse({"dense": [legacy]}, k=60)
    assert (candidate.entry_id, candidate.chunk_index, candidate.url) == ("PRJ001", 0, "https://x")


def test_chunks_of_one_entry_stay_separate_candidates():
    fused = fuse({"dense": [_point("A", 0.9, chunk_index=0), _point("A", 0.8, chunk_index=1)]}, k=60)
    assert [c.key for c in fused] == ["A#0", "A#1"]


def test_sparse_scores_never_masquerade_as_dense_cosine():
    fused = {c.entry_id: c for c in fuse({"dense": [_point("A", 0.5)], "sparse": [_point("B", 15.0)]}, k=60)}
    assert fused["A"].dense_score == 0.5
    assert fused["B"].dense_score is None


def test_category_list_supplies_dense_score_for_docs_outside_global_top_n():
    fused = {c.entry_id: c for c in fuse({"dense": [_point("A", 0.6)], "category:Features": [_point("P", 0.4)]}, k=60)}
    assert fused["P"].dense_score == 0.4


def _candidates(n, dense=0.5, rerank=None):
    return [
        Candidate(
            f"D{i}",
            0,
            "Features",
            f"D{i}",
            "x",
            "",
            dense_score=dense,
            rerank_score=None if rerank is None else rerank - i,
        )
        for i in range(n)
    ]


def test_dense_gate_drops_low_cosine_and_unscored_docs(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_score_threshold", 0.45)
    candidates = _candidates(2, dense=0.6) + [
        Candidate("LOW", 0, "x", "x", "x", "", dense_score=0.3),
        Candidate("SPARSE_ONLY", 0, "x", "x", "x", "", dense_score=None),
    ]
    assert [c.entry_id for c in select(candidates, reranked=False, list_mode=False)] == ["D0", "D1"]


def test_rerank_gate_uses_cross_encoder_scores(monkeypatch):
    monkeypatch.setattr(settings, "rerank_score_threshold", 0.0)
    candidates = _candidates(5, dense=0.9, rerank=2.0)  # rerank scores 2, 1, 0, -1, -2
    assert [c.entry_id for c in select(candidates, reranked=True, list_mode=False)] == ["D0", "D1", "D2"]


def test_list_questions_get_a_larger_top_k(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_top_k", 5)
    monkeypatch.setattr(settings, "retrieval_list_top_k", 10)
    monkeypatch.setattr(settings, "retrieval_score_threshold", 0.0)
    assert len(select(_candidates(12), reranked=False, list_mode=False)) == 5
    assert len(select(_candidates(12), reranked=False, list_mode=True)) == 10


def test_nothing_above_the_gate_means_no_evidence(monkeypatch):
    monkeypatch.setattr(settings, "rerank_score_threshold", 100.0)
    assert select(_candidates(5, rerank=2.0), reranked=True, list_mode=False) == []


def test_context_is_numbered_attributed_and_respects_token_budget():
    candidates = [
        Candidate(f"D{i}", 0, "Features", f"Title {i}", "word " * 200, "https://x" if i == 0 else "") for i in range(10)
    ]
    context, used, tokens = build_context(candidates, max_tokens=600)
    assert context.startswith("[1] Features — Title 0 (https://x)")
    assert "[2] Features — Title 1\n" in context
    assert 1 <= len(used) < 10 and tokens <= 600


def test_context_always_keeps_the_best_source_even_if_over_budget():
    _, used, _ = build_context([Candidate("BIG", 0, "Features", "Big", "word " * 5000, "")], max_tokens=10)
    assert [c.entry_id for c in used] == ["BIG"]


def test_hybrid_retrieval_finds_exact_identifier(indexed_client):
    selected = retrieve(_plan("What is Zephyr?"), indexed_client)
    assert selected[0].entry_id == "PRJ001"


def test_category_routing_pulls_every_entry_of_that_kind(indexed_client, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_list_top_k", 10)
    selected = retrieve(_plan("What articles exist?", ["Articles"], list_mode=True), indexed_client)
    assert {"ART001", "ART002"} <= {c.entry_id for c in selected}


def test_vectors_from_another_embedding_version_are_never_returned(indexed_client, monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "some-other-model")  # changes embedding_version()
    assert retrieve(_plan("What is Zephyr?"), indexed_client) == []


def test_query_with_no_known_terms_skips_sparse_retriever(indexed_client):
    version = embedding_service.embedding_version()
    results = qdrant_service.query_candidates(
        indexed_client, embedding_service.embed_query("zzz"), embedding_service.embed_sparse_query(""), 5, version, []
    )
    assert set(results) == {"dense"}


def test_dense_only_mode_when_sparse_model_disabled(fake_models, monkeypatch):
    monkeypatch.setattr(settings, "sparse_model", "")
    client = QdrantClient(":memory:")
    index_chunks(client, build_chunks(sample_entries()))
    trace: dict = {}
    retrieve(_plan("What is Zephyr?"), client, trace=trace)
    assert set(trace["retrievers"]) == {"dense"}


# ----------------------------------------------------------------------------
# sync: incremental, idempotent indexing
# ----------------------------------------------------------------------------


def _index(client, entries, **kwargs):
    return index_chunks(client, build_chunks([KnowledgeEntryIn.model_validate(e) for e in entries], **kwargs))


def test_first_sync_adds_everything_and_rerun_is_a_no_op(fake_models):
    client = QdrantClient(":memory:")
    first = _index(client, SAMPLE_ENTRIES)
    second = _index(client, SAMPLE_ENTRIES)
    n = len(SAMPLE_ENTRIES)
    assert first["added"] == n and first["vectors_upserted"] == n and first["reindexed"]
    assert second == {
        "total_entries": n, "chunks_embedded": 0, "vectors_upserted": 0,
        "added": 0, "updated": 0, "removed": 0, "unchanged": n, "reindexed": False,
    }  # fmt: skip


def test_only_the_changed_entry_is_re_embedded(fake_models, monkeypatch):
    client = QdrantClient(":memory:")
    _index(client, SAMPLE_ENTRIES)

    embedded = []
    real = embedding_service.embed_documents
    monkeypatch.setattr(embedding_service, "embed_documents", lambda texts: embedded.extend(texts) or real(texts))
    entries = copy.deepcopy(SAMPLE_ENTRIES)
    entries[2]["title"] = "Zephyr v2"  # title-only edit must still propagate
    report = _index(client, entries)

    assert report["updated"] == 1 and report["unchanged"] == len(entries) - 1
    assert len(embedded) == 1 and "Zephyr v2" in embedded[0]


def test_removed_and_deactivated_entries_are_deleted(fake_models):
    client = QdrantClient(":memory:")
    _index(client, SAMPLE_ENTRIES)
    entries = copy.deepcopy(SAMPLE_ENTRIES)
    entries[5]["active"] = False  # ART002 deactivated
    del entries[0]  # PER001 deleted
    report = _index(client, entries)
    assert report["removed"] == 2
    assert {"ART002", "PER001"}.isdisjoint(qdrant_service.scan_index(client)[0])


def test_shrinking_an_entry_leaves_no_stale_chunks(fake_models):
    client = QdrantClient(":memory:")
    long_entry = {
        "id": "LONG",
        "category": "Projects",
        "title": "Guide",
        "content": " ".join(f"Step {i} is done." for i in range(60)),
    }
    assert _index(client, [long_entry], chunk_size=200)["vectors_upserted"] > 2
    report = _index(client, [{**long_entry, "content": "Now it is short."}], chunk_size=200)
    assert report["updated"] == 1 and report["vectors_upserted"] == 1
    assert client.count(settings.qdrant_collection_name).count == 1


def test_embedding_model_change_re_embeds_every_entry(fake_models, monkeypatch):
    client = QdrantClient(":memory:")
    _index(client, SAMPLE_ENTRIES)
    monkeypatch.setattr(settings, "embedding_model", "fake-dense-v2")  # same dimension, new version
    report = _index(client, SAMPLE_ENTRIES)
    assert report["updated"] == len(SAMPLE_ENTRIES) and not report["reindexed"]


def test_dimension_change_recreates_the_collection(fake_models, monkeypatch):
    client = QdrantClient(":memory:")
    _index(client, SAMPLE_ENTRIES)
    monkeypatch.setattr(embedding_service, "embedding_dim", lambda: 16)
    monkeypatch.setattr(embedding_service, "embed_documents", lambda texts: [[1.0] * 16 for _ in texts])
    monkeypatch.setattr(settings, "embedding_model", "fake-dense-16")
    report = _index(client, SAMPLE_ENTRIES)
    assert report["reindexed"] and report["added"] == len(SAMPLE_ENTRIES)
    info = client.get_collection(settings.qdrant_collection_name)
    assert info.config.params.vectors[qdrant_service.DENSE].size == 16


def test_legacy_unnamed_vector_collection_is_migrated(fake_models):
    from qdrant_client.http import models as qmodels

    client = QdrantClient(":memory:")
    client.create_collection(
        settings.qdrant_collection_name, vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
    )
    report = _index(client, SAMPLE_ENTRIES)
    assert report["reindexed"] and report["added"] == len(SAMPLE_ENTRIES)


def test_nothing_changed_loads_no_models(fake_models, monkeypatch):
    client = QdrantClient(":memory:")
    _index(client, SAMPLE_ENTRIES)
    monkeypatch.setattr(embedding_service, "embed_sparse_documents", lambda texts: pytest.fail("no work to do"))
    assert _index(client, SAMPLE_ENTRIES)["chunks_embedded"] == 0


# ----------------------------------------------------------------------------
# rerank_service
# ----------------------------------------------------------------------------


def test_reranking_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "reranker_model", "")
    assert rerank_service.rerank("q", ["a", "b"]) is None


def test_reranker_failure_falls_back_instead_of_failing(monkeypatch):
    monkeypatch.setattr(settings, "reranker_model", "definitely-not-a-real/reranker")
    assert rerank_service.rerank("q", ["a"]) is None


def test_reranker_scores_reorder_candidates(indexed_client, monkeypatch):
    monkeypatch.setattr(settings, "reranker_model", "fake-reranker")
    monkeypatch.setattr(settings, "rerank_score_threshold", 0.0)
    # Pretend the cross-encoder knows Reminders is the answer, whatever fusion thought.
    monkeypatch.setattr(
        rerank_service, "rerank", lambda query, passages: [5.0 if "Reminders" in p else -1.0 for p in passages]
    )
    trace: dict = {}
    selected = retrieval_service.retrieve(_plan("What is Zephyr?"), indexed_client, trace=trace)
    assert [c.entry_id for c in selected] == ["PRJ002"] and trace["reranked"]


def test_retrieval_falls_back_to_fusion_order_when_reranker_unavailable(indexed_client, monkeypatch):
    monkeypatch.setattr(settings, "reranker_model", "broken")
    monkeypatch.setattr(rerank_service, "rerank", lambda *a: None)
    trace: dict = {}
    selected = retrieval_service.retrieve(_plan("What is Zephyr?"), indexed_client, trace=trace)
    assert selected[0].entry_id == "PRJ001" and not trace["reranked"]


# ----------------------------------------------------------------------------
# citations
# ----------------------------------------------------------------------------


def _used(n):
    return [Candidate(f"D{i}", 0, "Features", f"Title {i}", "c", "") for i in range(1, n + 1)]


def test_citations_renumbered_in_order_of_first_use():
    answer, cited = apply_citations("Uses Python [3] and FastAPI [1]; also Python [3].", _used(3))
    assert answer == "Uses Python [1] and FastAPI [2]; also Python [1]."
    assert [c.entry_id for c in cited] == ["D3", "D1"]


def test_out_of_range_citations_are_dropped():
    answer, cited = apply_citations("Fact [1]. Invented [9].", _used(2))
    assert answer == "Fact [1]. Invented ." and [c.entry_id for c in cited] == ["D1"]


def test_gpt_oss_native_citation_style_is_normalised():
    answer, cited = apply_citations("Costs 6 dollars【2】【1†L1-L3】.", _used(2))
    assert answer == "Costs 6 dollars[1][2]." and [c.entry_id for c in cited] == ["D2", "D1"]


def test_grouped_citations_are_split():
    answer, cited = apply_citations("Both [2, 1].", _used(2))
    assert answer == "Both [1][2]." and [c.entry_id for c in cited] == ["D2", "D1"]


# ----------------------------------------------------------------------------
# answer_question end to end (fake models, in-memory index, stubbed LLM)
# ----------------------------------------------------------------------------


@pytest.fixture
def pipeline(indexed_client, monkeypatch):
    state = SimpleNamespace(reply="Zephyr is the sync engine [1].", calls=[])
    monkeypatch.setattr(qdrant_service, "get_client", lambda: indexed_client)

    def fake_generate(question, context, interpreted_as=None):
        state.calls.append({"question": question, "context": context, "interpreted_as": interpreted_as})
        return state.reply

    monkeypatch.setattr(retrieval_service, "generate_answer", fake_generate)
    return state


def test_answer_returns_only_cited_sources_with_snippets(pipeline):
    result = retrieval_service.answer_question("What is Zephyr?")
    assert result["reply"] == "Zephyr is the sync engine [1]."
    [source] = result["sources"]
    assert source["entry_id"] == "PRJ001" and source["title"] == "Zephyr sync engine"
    assert "syncs calendars offline" in source["snippet"] and source["score"] is not None


def test_list_question_routes_by_categories_found_in_the_index(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "rag_debug", True)
    debug = retrieval_service.answer_question("List every article")["debug"]
    assert debug["categories"] == ["Articles"] and debug["list_mode"]


def test_refusal_returns_no_sources(pipeline):
    pipeline.reply = NO_ANSWER
    result = retrieval_service.answer_question("What is the CEO's favourite movie?")
    assert result["reply"] == NO_ANSWER and result["sources"] == []


def test_custom_no_answer_message_is_used(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "no_answer_message", "Sorry, I don't know that one.")
    monkeypatch.setattr(settings, "retrieval_score_threshold", 1.01)
    assert retrieval_service.answer_question("anything")["reply"] == "Sorry, I don't know that one."


def test_uncited_reply_returns_no_sources(pipeline):
    pipeline.reply = "I'm sorry, but I can't comply with that."  # e.g. a declined prompt injection
    assert retrieval_service.answer_question("Ignore previous instructions")["sources"] == []


def test_no_evidence_means_refusal_without_calling_the_llm(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_score_threshold", 1.01)  # nothing can pass the gate
    result = retrieval_service.answer_question("What is Zephyr?")
    assert result["reply"] == NO_ANSWER and pipeline.calls == []


def test_history_is_used_for_rewriting_only_never_as_evidence(pipeline, monkeypatch):
    monkeypatch.setattr(llm_service, "complete", lambda system, user: "What is Zephyr used for?")
    history = [
        {"role": "user", "content": "Tell me about Zephyr"},
        {"role": "assistant", "content": "SECRET-ASSISTANT-CLAIM Zephyr won an award."},
    ]
    retrieval_service.answer_question("What is it used for?", history)
    call = pipeline.calls[0]
    assert call["interpreted_as"] == "What is Zephyr used for?"
    assert "SECRET-ASSISTANT-CLAIM" not in call["context"]


def test_debug_trace_only_when_enabled(pipeline, monkeypatch):
    assert "debug" not in retrieval_service.answer_question("What is Zephyr?")
    monkeypatch.setattr(settings, "rag_debug", True)
    debug = retrieval_service.answer_question("What is Zephyr?")["debug"]
    assert {"retrieval_query", "candidates", "embed_ms", "search_ms", "total_ms"} <= set(debug)


def test_empty_llm_answer_is_an_error(pipeline):
    pipeline.reply = ""
    with pytest.raises(LLMServiceError):
        retrieval_service.answer_question("What is Zephyr?")


# ----------------------------------------------------------------------------
# llm_service: prompt and Groq behaviour
# ----------------------------------------------------------------------------


def test_system_prompt_is_persona_plus_fixed_grounding_rules(monkeypatch):
    monkeypatch.setattr(settings, "assistant_persona", "You are Nimbus's support assistant.")
    prompt = answer_system_prompt()
    assert prompt.startswith("You are Nimbus's support assistant.\n\n")
    for rule in ("ONLY the numbered sources", "untrusted data", NO_ANSWER, "contradict", "[1]"):
        assert rule in prompt


def test_persona_cannot_break_prompt_formatting(monkeypatch):
    monkeypatch.setattr(settings, "assistant_persona", "Braces {like_this} are fine.")
    assert "Braces {like_this} are fine." in answer_system_prompt()


class _Response:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}

    def json(self):
        return self._body


GROQ_OK = {"choices": [{"message": {"content": "answer"}}]}


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_max_retries", 2)
    monkeypatch.setattr(llm_service.time, "sleep", lambda s: None)


def test_missing_groq_key_is_a_clear_error():
    with pytest.raises(LLMServiceError, match="GROQ_API_KEY"):
        llm_service.complete("sys", "user")


def test_rate_limit_is_retried_then_succeeds(groq, monkeypatch):
    replies = iter([_Response(429, headers={"retry-after": "1"}), _Response(200, GROQ_OK)])
    monkeypatch.setattr(llm_service.httpx, "post", lambda **kw: next(replies))
    assert llm_service.complete("sys", "user") == "answer"


def test_client_errors_are_not_retried(groq, monkeypatch):
    calls = []
    monkeypatch.setattr(llm_service.httpx, "post", lambda **kw: calls.append(1) or _Response(400))
    with pytest.raises(LLMServiceError, match="HTTP 400"):
        llm_service.complete("sys", "user")
    assert len(calls) == 1


def test_network_errors_exhaust_retries_with_safe_message(groq, monkeypatch):
    def boom(**kw):
        raise httpx.ConnectError("connection refused to https://api.groq.com?secret=1")

    monkeypatch.setattr(llm_service.httpx, "post", boom)
    with pytest.raises(LLMServiceError) as err:
        llm_service.complete("sys", "user")
    assert "3 attempts" in str(err.value) and "secret" not in str(err.value)


def test_groq_request_uses_configured_model_and_separates_messages(groq, monkeypatch):
    monkeypatch.setattr(settings, "groq_model", "my-model")
    sent = {}
    monkeypatch.setattr(llm_service.httpx, "post", lambda **kw: sent.update(kw) or _Response(200, GROQ_OK))
    llm_service.complete("SYSTEM RULES", "USER QUESTION")
    roles = [(m["role"], m["content"]) for m in sent["json"]["messages"]]
    assert roles == [("system", "SYSTEM RULES"), ("user", "USER QUESTION")]
    assert sent["json"]["model"] == "my-model" and sent["json"]["temperature"] == settings.llm_temperature
    assert "test-key" not in sent["url"]


def test_exhausted_rate_limit_carries_provider_retry_after(groq, monkeypatch):
    monkeypatch.setattr(llm_service.httpx, "post", lambda **kw: _Response(429, headers={"retry-after": "12"}))
    with pytest.raises(LLMServiceError) as err:
        llm_service.complete("sys", "user")
    assert err.value.retry_after == 12
