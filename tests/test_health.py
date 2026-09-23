"""Tests for the /health endpoint with model status reporting."""

import os
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import reset_engine


@pytest.fixture(autouse=True)
def _clean_engine():
    """Reset the singleton engine before each test."""
    reset_engine()
    # Ensure mock mode is set for tests (no real model available)
    os.environ["CORRECTION_MOCK_MODE"] = "true"
    yield
    reset_engine()
    os.environ.pop("CORRECTION_MOCK_MODE", None)


@pytest.fixture
def client():
    """Create a fresh TestClient that triggers the lifespan."""
    # Import here so the module-level state is reset first
    from app.api.main import app
    with TestClient(app) as c:
        yield c


def test_health_check_ok(client):
    """Health endpoint reports ok when engine is initialized in mock mode."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True
    assert data["model_loaded"] is True  # mock_mode counts as "loaded"


def test_health_reports_model_id(client):
    """Health endpoint includes the configured model identifier."""
    response = client.get("/health")
    data = response.json()
    assert "model_id" in data
    assert isinstance(data["model_id"], str)
