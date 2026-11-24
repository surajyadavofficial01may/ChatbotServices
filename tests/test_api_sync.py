"""API tests: the automation webhook for synchronization."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.sync import SyncReport

client = TestClient(app)

OK_REPORT = SyncReport(success=True, total_entries=35, added=1, unchanged=34, vectors_upserted=1, chunks_embedded=1)


@pytest.fixture
def webhook_secret(monkeypatch):
    monkeypatch.setattr(settings, "webhook_secret", "hook-secret")
    monkeypatch.setattr(settings, "sync_enabled", True)


def test_webhook_rejects_missing_secret(webhook_secret):
    response = client.post("/api/v1/webhook/sync")
    assert response.status_code == 401
    assert response.json() == {"success": False, "error": "Invalid webhook secret."}


def test_webhook_rejects_wrong_secret(webhook_secret):
    assert client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "nope"}).status_code == 401


def test_webhook_rejects_non_ascii_secret_without_crashing(webhook_secret):
    response = client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "naïve".encode()})
    assert response.status_code == 401


def test_webhook_unavailable_when_secret_not_configured():
    assert client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "anything"}).status_code == 503


def test_webhook_refused_when_sync_disabled(webhook_secret, monkeypatch):
    monkeypatch.setattr(settings, "sync_enabled", False)
    response = client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "hook-secret"})
    assert response.status_code == 403


def test_webhook_runs_sync_with_correct_secret_and_ignores_the_body(webhook_secret):
    with patch("app.services.sync_service.knowledge_sync.run", new=AsyncMock(return_value=OK_REPORT)) as mock_sync:
        response = client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "hook-secret"}, json={"x": 1})
    assert response.status_code == 200 and response.json()["added"] == 1
    mock_sync.assert_awaited_once_with()


def test_admin_password_does_not_open_the_webhook(webhook_secret, monkeypatch):
    monkeypatch.setattr(settings, "admin_password", "admin-secret")
    assert client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "admin-secret"}).status_code == 401
