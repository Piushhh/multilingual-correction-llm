from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_id" in data
    assert "device" in data

def test_correct_mock():
    # Use mock mode to bypass actual model loading for tests
    req = {
        "text": "teh cat",
        "language": "en",
        "domain": "general",
        "mock_mode": True
    }
    response = client.post("/correct", json=req)
    assert response.status_code == 200
    data = response.json()
    assert "the cat" in data["corrected_text"]
    assert "changes" in data
    assert "metadata" in data

def test_correct_batch_mock():
    req = {
        "requests": [
            {
                "text": "teh cat",
                "language": "en",
                "domain": "general",
                "mock_mode": True
            },
            {
                "text": "he have",
                "language": "en",
                "domain": "general",
                "mock_mode": True
            }
        ]
    }
    response = client.post("/correct/batch", json=req)
    assert response.status_code == 200
    data = response.json()
    assert len(data["responses"]) == 2
    assert "the cat" in data["responses"][0]["corrected_text"]
