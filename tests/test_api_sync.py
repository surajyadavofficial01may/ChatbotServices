"""API tests: webhook and admin synchronization endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.sync import SyncReport

# ----------------------------------------------------------------------------
# /admin/sync
# ----------------------------------------------------------------------------

client = TestClient(app)


def test_sync_requires_token_when_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", "")
    response = client.post("/api/v1/admin/sync")
    assert response.status_code == 503


def test_sync_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", "secret-token")
    response = client.post("/api/v1/admin/sync", headers={"x-admin-token": "wrong"})
    assert response.status_code == 401


def test_sync_accepts_correct_token(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", "secret-token")
    monkeypatch.setattr(settings, "supabase_url", "")  # force a deterministic failure, no live network call
    response = client.post("/api/v1/admin/sync", headers={"x-admin-token": "secret-token"})
    assert response.status_code == 200
    assert response.json()["success"] is False  # no Supabase credentials configured in this test


# ----------------------------------------------------------------------------
# /webhook/sync
# ----------------------------------------------------------------------------

OK_REPORT = SyncReport(success=True, added=1, unchanged=34)


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


def test_webhook_unavailable_when_secret_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "webhook_secret", "")
    assert client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "anything"}).status_code == 503


def test_webhook_refused_when_sync_disabled(webhook_secret, monkeypatch):
    monkeypatch.setattr(settings, "sync_enabled", False)
    response = client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "hook-secret"})
    assert response.status_code == 403


def test_webhook_runs_sync_with_correct_secret(webhook_secret):
    with patch("app.api.routes_sync.run_sync", new=AsyncMock(return_value=OK_REPORT)) as mock_sync:
        response = client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "hook-secret"}, json={"x": 1})
    assert response.status_code == 200 and response.json()["added"] == 1
    mock_sync.assert_awaited_once()


def test_admin_token_does_not_open_the_webhook(webhook_secret, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", "admin-secret")
    assert client.post("/api/v1/webhook/sync", headers={"X-Webhook-Secret": "admin-secret"}).status_code == 401


def test_admin_sync_refused_when_sync_disabled(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", "admin-secret")
    monkeypatch.setattr(settings, "sync_enabled", False)
    assert client.post("/api/v1/admin/sync", headers={"x-admin-token": "admin-secret"}).status_code == 403
