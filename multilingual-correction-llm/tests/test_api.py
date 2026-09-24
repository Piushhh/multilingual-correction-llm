"""
API endpoint tests for correction and model info routes.

All tests run with CORRECTION_MOCK_MODE=true so no real model is needed.
"""

import os
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import reset_engine


@pytest.fixture(autouse=True)
def _clean_engine():
    """Reset the singleton engine before each test."""
    reset_engine()
    os.environ["CORRECTION_MOCK_MODE"] = "true"
    yield
    reset_engine()
    os.environ.pop("CORRECTION_MOCK_MODE", None)


@pytest.fixture
def client():
    """Create a fresh TestClient with lifespan."""
    from app.api.main import app
    with TestClient(app) as c:
        yield c


# ── /model/info ──────────────────────────────────────────────────────

def test_model_info(client):
    """Model info endpoint returns expected fields."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_id" in data
    assert "device" in data
    assert "loaded" in data
    assert "mock_mode" in data
    assert "baseline" in data
    assert data["mock_mode"] is True


def test_model_info_baseline_flag(client):
    """Model info correctly flags the baseline model."""
    response = client.get("/model/info")
    data = response.json()
    # Default config uses gemma, so baseline should be True
    assert data["baseline"] is True
    assert "baseline" in data["note"].lower()


# ── POST /correct ────────────────────────────────────────────────────

def test_correct_mock(client):
    """Correction endpoint returns structured output in mock mode."""
    req = {
        "text": "teh cat",
        "language": "en",
        "domain": "general",
    }
    response = client.post("/correct", json=req)
    assert response.status_code == 200
    data = response.json()
    assert "the cat" in data["corrected_text"]
    assert "changes" in data
    assert "metadata" in data
    assert data["metadata"]["language"] == "en"


def test_correct_empty_text(client):
    """Correction handles empty text gracefully."""
    req = {"text": "", "language": "en", "domain": "general"}
    response = client.post("/correct", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["corrected_text"] == ""


def test_correct_hindi_text(client):
    """Correction handles Hindi text input."""
    req = {"text": "यह एक परीक्षण है", "language": "hi", "domain": "general"}
    response = client.post("/correct", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["language"] == "hi"


# ── POST /correct/batch ─────────────────────────────────────────────

def test_correct_batch_mock(client):
    """Batch correction returns structured output for multiple inputs."""
    req = {
        "requests": [
            {"text": "teh cat", "language": "en", "domain": "general"},
            {"text": "he have", "language": "en", "domain": "general"},
        ]
    }
    response = client.post("/correct/batch", json=req)
    assert response.status_code == 200
    data = response.json()
    assert len(data["responses"]) == 2
    assert "the cat" in data["responses"][0]["corrected_text"]


def test_batch_exceeds_limit(client):
    """Batch endpoint rejects requests exceeding the 50-item limit."""
    req = {
        "requests": [
            {"text": f"text {i}", "language": "en", "domain": "general"}
            for i in range(51)
        ]
    }
    response = client.post("/correct/batch", json=req)
    assert response.status_code == 400
    assert "limit" in response.json()["detail"].lower()


# ── Schema validation ───────────────────────────────────────────────

def test_correct_missing_text(client):
    """Correction endpoint requires the 'text' field."""
    response = client.post("/correct", json={"language": "en"})
    assert response.status_code == 422  # Pydantic validation error
