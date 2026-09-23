import pytest
from src.integration.pipeline import IntegrationPipeline
from src.integration.adapters.ocr_adapter import OCRAdapter, DomainAdapter
from src.correction.inference import CorrectionEngine

@pytest.fixture
def mock_engine():
    engine = CorrectionEngine({"model_name": "mock"}, mock_mode=True)
    engine.load_model()
    return engine

def test_integration_pipeline(mock_engine):
    pipeline = IntegrationPipeline(mock_engine)
    
    raw_ocr = {
        "document_id": "doc_123",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"block_id": "b1", "text": "teh cat", "bbox": [0,0,10,10], "confidence": 0.9}
                ]
            }
        ]
    }
    
    raw_domain = {
        "domain": "general",
        "confidence": 0.95
    }
    
    result = pipeline.process_document(raw_ocr, raw_domain)
    
    assert result.document_id == "doc_123"
    assert len(result.pages) == 1
    assert len(result.pages[0].blocks) == 1
    
    block = result.pages[0].blocks[0]
    assert block.block_id == "b1"
    assert block.original_text == "teh cat"
    assert "the cat" in block.corrected_text
    assert block.domain == "general"
