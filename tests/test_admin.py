"""Admin studio tests: authentication, schema, entry CRUD, bulk import, sync, and the /dataentry page."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.api import dependencies
from app.core.config import settings
from app.main import app
from app.models.knowledge import KnowledgeEntry, KnowledgeEntryIn
from app.services import sync_service
from app.services.supabase_repository import DuplicateEntryError, SupabaseRepository
from app.services.sync_service import KnowledgeSyncService
from tests.conftest import SAMPLE_ENTRIES, sample_entries

PASSWORD = "correct horse battery staple"
SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_entries.json"


class FakeRepository:
    """In-memory stand-in for SupabaseRepository."""

    def __init__(self, entries=()):
        self.rows = {e.id: self._stored(e) for e in entries}

    @staticmethod
    def _stored(entry):
        return KnowledgeEntry(**entry.model_dump(), updated_at=datetime.now(UTC))

    async def list_entries(self, active_only=False):
        return [e for e in self.rows.values() if e.active or not active_only]

    async def create_entry(self, entry):
        if entry.id in self.rows:
            raise DuplicateEntryError("An entry with this ID already exists.")
        self.rows[entry.id] = self._stored(entry)
        return self.rows[entry.id]

    async def update_entry(self, entry_id, entry):
        if entry_id not in self.rows:
            return None
        self.rows[entry_id] = self._stored(entry.model_copy(update={"id": entry_id}))
        return self.rows[entry_id]

    async def delete_entry(self, entry_id):
        return self.rows.pop(entry_id, None) is not None

    async def upsert_entries(self, entries):
        for entry in entries:
            self.rows[entry.id] = self._stored(entry)
        return len(entries)


@pytest.fixture
def repo(monkeypatch):
    fake = FakeRepository(sample_entries())
    monkeypatch.setattr(sync_service.knowledge_sync, "repository", fake)
    return fake


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    client = TestClient(app)
    assert client.post("/api/v1/admin/auth", json={"password": PASSWORD}).status_code == 200
    return client


# ----------------------------------------------------------------------------
# authentication
# ----------------------------------------------------------------------------

PROTECTED = [
    ("GET", "/api/v1/admin/schema"),
    ("GET", "/api/v1/admin/entries"),
    ("POST", "/api/v1/admin/entries"),
    ("PUT", "/api/v1/admin/entries/PER001"),
    ("DELETE", "/api/v1/admin/entries/PER001"),
    ("POST", "/api/v1/admin/entries/bulk"),
    ("POST", "/api/v1/admin/sync"),
    ("GET", "/api/v1/admin/sync"),
]


@pytest.mark.parametrize("method, path", PROTECTED)
def test_every_admin_route_requires_a_session(method, path, monkeypatch, repo):
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    response = TestClient(app).request(method, path, json={})
    assert response.status_code == 401
    assert response.json() == {"success": False, "error": "Not signed in."}


def test_admin_is_unavailable_until_a_password_is_configured():
    client = TestClient(app)
    assert client.post("/api/v1/admin/auth", json={"password": "anything"}).status_code == 503
    assert client.get("/api/v1/admin/schema").status_code == 503


def test_wrong_password_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    response = TestClient(app).post("/api/v1/admin/auth", json={"password": "wrong"})
    assert response.status_code == 401 and "set-cookie" not in response.headers


def test_login_sets_a_hardened_session_cookie(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    response = TestClient(app).post("/api/v1/admin/auth", json={"password": PASSWORD})
    cookie = response.headers["set-cookie"].lower()
    assert response.json() == {"success": True, "expires_in": dependencies.SESSION_SECONDS}
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/api/v1/admin" in cookie
    assert PASSWORD.lower() not in cookie


def test_session_cookie_is_secure_in_production(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    monkeypatch.setattr(settings, "app_env", "production")
    response = TestClient(app).post("/api/v1/admin/auth", json={"password": PASSWORD})
    assert "secure" in response.headers["set-cookie"].lower()


def test_valid_session_opens_admin_routes(admin, repo):
    assert admin.get("/api/v1/admin/entries").status_code == 200


def test_forged_and_expired_tokens_are_rejected(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    token = dependencies.issue_session_token()
    expires, signature = token.split(".")
    assert dependencies.session_is_valid(token)
    assert not dependencies.session_is_valid(f"{int(expires) + 999999}.{signature}")  # extended expiry
    assert not dependencies.session_is_valid(f"{expires}.{'0' * 64}")
    assert not dependencies.session_is_valid("garbage")
    assert not dependencies.session_is_valid(dependencies.issue_session_token(now=0))  # long expired


def test_changing_the_password_ends_existing_sessions(admin, monkeypatch, repo):
    monkeypatch.setattr(settings, "admin_password", "a brand new password")
    assert admin.get("/api/v1/admin/entries").status_code == 401


def test_login_attempts_are_rate_limited(monkeypatch):
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    client = TestClient(app)
    codes = [client.post("/api/v1/admin/auth", json={"password": "guess"}).status_code for _ in range(6)]
    assert codes == [401] * 5 + [429]


def test_logout_clears_the_session(admin, repo):
    response = admin.post("/api/v1/admin/logout")
    assert response.status_code == 200 and 'admin_session=""' in response.headers["set-cookie"]
    assert admin.get("/api/v1/admin/entries").status_code == 401


# ----------------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------------


def test_schema_endpoint_describes_the_form(admin):
    schema = admin.get("/api/v1/admin/schema").json()
    assert schema == KnowledgeEntryIn.model_json_schema()
    assert {p.get("ui") for p in schema["properties"].values()} == {"id", "tabs", None, "textarea", "url", "order"}


def test_entries_serialize_with_timestamps(admin, repo):
    [first, *_] = admin.get("/api/v1/admin/entries").json()
    assert set(first) == {
        "id", "category", "title", "content", "url", "active", "sort_order", "updated_at"
    }  # fmt: skip
    assert first["updated_at"].endswith(("Z", "+00:00"))


# ----------------------------------------------------------------------------
# CRUD
# ----------------------------------------------------------------------------


def test_create_update_delete_round_trip(admin, repo):
    created = admin.post(
        "/api/v1/admin/entries", json={"id": "PRJ009", "category": "Projects", "title": "New", "content": "Body"}
    )
    assert created.status_code == 201
    entry_id = created.json()["id"]
    assert entry_id == "PRJ009" and entry_id in repo.rows and created.json()["url"] == ""

    updated = admin.put(
        f"/api/v1/admin/entries/{entry_id}",
        json={"id": "X", "category": "Projects", "title": "Renamed", "content": "Body"},
    )
    assert updated.status_code == 200 and updated.json()["title"] == "Renamed" and updated.json()["id"] == entry_id

    assert admin.delete(f"/api/v1/admin/entries/{entry_id}").json() == {"success": True}
    assert entry_id not in repo.rows


def test_duplicate_id_is_a_409(admin, repo):
    response = admin.post(
        "/api/v1/admin/entries", json={"id": "PER001", "category": "Personal_Details", "content": "C"}
    )
    assert response.status_code == 409 and "already exists" in response.json()["error"]


def test_missing_entries_are_404(admin, repo):
    assert (
        admin.put("/api/v1/admin/entries/NOPE", json={"id": "NOPE", "category": "FAQs", "content": "C"}).status_code
        == 404
    )
    assert admin.delete("/api/v1/admin/entries/NOPE").status_code == 404


def test_invalid_entries_report_each_field(admin, repo):
    response = admin.post(
        "/api/v1/admin/entries", json={"id": "PRJ010", "category": "General", "content": "C", "url": "ftp://x"}
    )
    fields = {d["field"] for d in response.json()["details"]}
    assert response.status_code == 422 and fields == {"category", "url"}


def test_malformed_entry_ids_in_the_path_are_rejected(admin, repo):
    assert admin.delete("/api/v1/admin/entries/bad%20id").status_code == 422


# ----------------------------------------------------------------------------
# bulk import
# ----------------------------------------------------------------------------


def test_bulk_import_upserts_and_merges_duplicate_ids(admin, repo):
    payload = [
        {"id": "PER001", "category": "Personal_Details", "title": "Updated title", "content": "Updated."},
        {"id": "PRJ010", "category": "Projects", "title": "Brand new", "content": "New row."},
        # A row copied from a per-tab view / spreadsheet: active as a string, extra columns ignored.
        {"id": "SK001", "category": "Technical_Skills", "content": "Go.", "active": "FALSE", "updated_at": "x"},
        {"id": "SK001", "category": "Technical_Skills", "title": "Go", "content": "Last one wins."},
    ]
    result = admin.post("/api/v1/admin/entries/bulk", json=payload).json()
    assert result == {"success": True, "upserted": 3, "duplicates_merged": 1}
    assert repo.rows["PER001"].title == "Updated title" and repo.rows["SK001"].content == "Last one wins."
    assert len(repo.rows) == len(SAMPLE_ENTRIES) + 2


def test_view_style_active_strings_are_understood(admin, repo):
    row = {"id": "SK002", "category": "Technical_Skills", "content": "Rust.", "active": "FALSE"}
    admin.post("/api/v1/admin/entries/bulk", json=[row])
    assert repo.rows["SK002"].active is False


def test_bulk_import_accepts_the_shipped_sample_file(admin, repo):
    rows = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    assert admin.post("/api/v1/admin/entries/bulk", json=rows).json()["upserted"] == len(rows)


def test_bulk_validation_errors_point_at_the_entry(admin, repo):
    rows = [{"id": "FAQ010", "category": "FAQs", "content": "ok"}, {"id": "FAQ011", "category": "FAQs", "title": "?"}]
    response = admin.post("/api/v1/admin/entries/bulk", json=rows)
    assert response.status_code == 422
    assert response.json()["details"] == [{"field": "1.content", "message": "Field required"}]


def test_bulk_import_is_capped(admin, repo):
    too_many = [{"id": "X", "category": "FAQs", "content": "c"}] * 2001
    assert admin.post("/api/v1/admin/entries/bulk", json=too_many).status_code == 422


# ----------------------------------------------------------------------------
# sync: mocked Supabase -> chunks -> fake embeddings -> in-memory Qdrant
# ----------------------------------------------------------------------------


def _mock_supabase(monkeypatch, rows):
    """A SupabaseRepository whose HTTP calls are answered from `rows`, honouring the active filter."""
    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(settings, "supabase_service_role_key", "sb_secret_test")

    def handler(request):
        active_only = request.url.params.get("active") == "eq.true"
        offset = int(request.url.params.get("offset", 0))
        return httpx.Response(200, json=[r for r in rows if r.get("active", True) or not active_only][offset:])

    return SupabaseRepository(transport=httpx.MockTransport(handler))


def test_admin_sync_indexes_supabase_into_qdrant(admin, monkeypatch, fake_models):
    rows = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr(sync_service.knowledge_sync, "repository", _mock_supabase(monkeypatch, rows))
    monkeypatch.setattr(sync_service.knowledge_sync, "client_factory", lambda: qdrant)

    report = admin.post("/api/v1/admin/sync").json()
    active = [r for r in rows if r.get("active", True)]
    assert report["success"] and report["total_entries"] == len(active) == report["added"]
    assert report["chunks_embedded"] == report["vectors_upserted"] == len(active) + 1  # one entry has 2 chunks
    assert report["duration_ms"] > 0 and report["finished_at"]
    assert qdrant.count(settings.qdrant_collection_name).count == report["vectors_upserted"]

    status = admin.get("/api/v1/admin/sync").json()
    assert status["running"] is False and status["stage"] == "idle"
    assert status["last_report"]["vectors_upserted"] == report["vectors_upserted"]

    again = admin.post("/api/v1/admin/sync").json()
    assert again["unchanged"] == len(active) and again["vectors_upserted"] == 0


def test_admin_sync_refused_when_disabled(admin, monkeypatch):
    monkeypatch.setattr(settings, "sync_enabled", False)
    assert admin.post("/api/v1/admin/sync").status_code == 403


def test_sync_reports_stages_in_order(fake_models):
    stages = []
    service = KnowledgeSyncService(FakeRepository(sample_entries()), client_factory=lambda: QdrantClient(":memory:"))
    service._set_stage = lambda stage: stages.append(stage)
    assert asyncio.run(service.run()).success
    assert stages == ["embedding", "upserting"] and service.status().stage == "idle"


def test_sync_failure_is_reported_not_raised():
    class Broken(FakeRepository):
        async def list_entries(self, active_only=False):
            raise RuntimeError("Supabase is down")

    service = KnowledgeSyncService(Broken())
    report = asyncio.run(service.run())
    assert not report.success and report.error == "Supabase is down" and service.last_report is report


def test_concurrent_sync_is_refused(fake_models):
    async def scenario():
        release = asyncio.Event()

        class Slow(FakeRepository):
            async def list_entries(self, active_only=False):
                await release.wait()
                return []

        service = KnowledgeSyncService(Slow(), client_factory=lambda: QdrantClient(":memory:"))
        first = asyncio.create_task(service.run())
        await asyncio.sleep(0)
        assert service.status().running and service.status().stage == "fetching"
        second = await service.run()
        release.set()
        return second, await first

    second, first = asyncio.run(scenario())
    assert second.error == "A synchronization is already in progress." and first.success


# ----------------------------------------------------------------------------
# /dataentry page
# ----------------------------------------------------------------------------


def test_dataentry_page_is_served_with_config_and_framing_protection(monkeypatch):
    monkeypatch.setattr(settings, "app_title", 'Acme <script>alert("x")</script>')
    response = TestClient(app).get("/dataentry")
    assert response.status_code == 200 and response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert 'data-api="/api/v1"' in response.text and "{{" not in response.text
    assert "<script>alert" not in response.text and "&lt;script&gt;" in response.text


def test_dataentry_script_is_served():
    response = TestClient(app).get("/dataentry/assets/dataentry.js")
    assert response.status_code == 200 and "javascript" in response.headers["content-type"]
