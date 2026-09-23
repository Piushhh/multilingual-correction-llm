import pytest
from src.correction.inference import CorrectionEngine

def test_inference_mock_mode():
    engine = CorrectionEngine({"model_name": "mock"})
    engine.load_model()  # Should fall back to mock
    
    request = {
        "text": "teh cat is sleeping",
        "language": "en",
        "domain": "general"
    }
    
    result = engine.correct(request)
    
    # In mock mode, we just check if it returns structured output
    assert "corrected_text" in result
    assert "changes" in result
    assert "metadata" in result
    
    assert result["metadata"]["language"] == "en"
    
    # In our specific mock implementation, "teh" gets replaced by "the"
    assert "the cat is sleeping" in result["corrected_text"]
    assert len(result["changes"]) > 0
    assert result["changes"][0]["category"] == "replacement"

def test_inference_empty_input():
    engine = CorrectionEngine({})
    engine.load_model()
    
    result = engine.correct({"text": ""})
    assert result["corrected_text"] == ""
    assert len(result["changes"]) == 0
