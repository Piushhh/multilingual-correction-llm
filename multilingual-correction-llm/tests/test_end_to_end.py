"""
Stage 11: End-to-End Pipeline Integration Tests.

Validates the full flow:
1. DocumentAI output structure -> DocumentAIAdapter -> OCRDocument.
2. OCRDocument + DomainClassification -> IntegrationPipeline -> CorrectedDocument.
3. FastAPI test client covering:
   - GET  /health (model loaded, status ok)
   - POST /correct (single block correction)
   - POST /correct/batch (multiple blocks)
   - GET  /model/info
   - POST /generate (Member 1 LLM endpoint)
4. Domain classifier + Terminology checking in the pipeline.
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.document_ai.schemas import DocumentAIOutput
from src.integration.adapters.document_ai_adapter import adapt_document_ai_to_ocr_document
from src.integration.pipeline import IntegrationPipeline
from src.correction.inference import CorrectionEngine
from app.api.dependencies import reset_engine


@pytest.fixture(autouse=True)
def setup_teardown():
    reset_engine()
    os.environ["CORRECTION_MOCK_MODE"] = "true"
    yield
    reset_engine()
    os.environ.pop("CORRECTION_MOCK_MODE", None)


def test_end_to_end_document_ai_to_correction():
    """
    Test full flow:
    Simulated DocumentAI output -> adapt_document_ai_to_ocr_document -> IntegrationPipeline.
    """
    # 1. Member 2 output structure (validated by DocumentAIOutput)
    raw_doc_ai = {
        "version": "1.0",
        "page_id": "doc_e2e_001",
        "language": "en",
        "language_confidence": 0.98,
        "domain": "deep_learning",
        "domain_confidence": 0.95,
        "regions": [
            {
                "text": "The transformer architechture uses self-attenshun.",
                "bbox": [10, 20, 300, 50],
                "confidence": 0.92,
                "terminology_flags": [
                    {
                        "word": "architechture",
                        "position": 16,
                        "status": "possible_ocr_error",
                        "closest_term": "architecture",
                        "similarity": 92.0,
                    }
                ],
            },
            {
                "text": "Gradient decsent updates weights.",
                "bbox": [10, 60, 250, 90],
                "confidence": 0.88,
                "terminology_flags": [],
            },
        ],
    }

    # Validate with Pydantic schema
    validated_output = DocumentAIOutput(**raw_doc_ai).model_dump()

    # 2. Adapt DocumentAI output to OCRDocument
    ocr_doc = adapt_document_ai_to_ocr_document(validated_output)
    assert ocr_doc.document_id == "doc_e2e_001"
    assert len(ocr_doc.pages) == 1
    assert len(ocr_doc.pages[0].blocks) == 2
    assert ocr_doc.pages[0].blocks[0].language == "English"
    assert ocr_doc.pages[0].blocks[0].domain == "deep_learning"

    # 3. Process with IntegrationPipeline using mock CorrectionEngine
    engine = CorrectionEngine(config={}, mock_mode=True)
    pipeline = IntegrationPipeline(correction_engine=engine)

    raw_domain = {"domain": "deep_learning", "confidence": 0.95}
    corrected_doc = pipeline.process_document(ocr_doc.model_dump(), raw_domain)

    assert corrected_doc.document_id == "doc_e2e_001"
    assert len(corrected_doc.pages) == 1
    assert len(corrected_doc.pages[0].blocks) == 2

    b1 = corrected_doc.pages[0].blocks[0]
    assert b1.block_id == "p1_r0"
    assert b1.original_text == "The transformer architechture uses self-attenshun."
    assert b1.domain == "deep_learning"
    assert b1.language == "English"


def test_api_full_suite():
    """
    Test all FastAPI endpoints in one integrated test:
    /health, /correct, /correct/batch, /model/info, /generate
    """
    from app.api.main import app

    with TestClient(app) as client:
        # 1. /health
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["mock_mode"] is True

        # 2. /model/info
        res = client.get("/model/info")
        assert res.status_code == 200
        assert "model_id" in res.json()

        # 3. /correct
        res = client.post(
            "/correct",
            json={
                "text": "This is a tst sentense.",
                "language": "English",
                "domain": "general",
            },
        )
        assert res.status_code == 200
        res_data = res.json()
        assert "corrected_text" in res_data
        assert "changes" in res_data

        # 4. /correct/batch
        res = client.post(
            "/correct/batch",
            json={
                "requests": [
                    {"text": "Sample one", "language": "en", "domain": "general"},
                    {"text": "Sample two", "language": "hi", "domain": "general"},
                ]
            },
        )
        assert res.status_code == 200
        assert len(res.json()["responses"]) == 2

        # 5. /generate (Member 1 LLM endpoint)
        res = client.post(
            "/generate",
            json={
                "prompt": "Deep learning models",
                "max_new_tokens": 15,
                "temperature": 0.7,
            },
        )
        assert res.status_code == 200
        gen_data = res.json()
        assert "text" in gen_data
        assert "model_id" in gen_data
        assert isinstance(gen_data["text"], str)
