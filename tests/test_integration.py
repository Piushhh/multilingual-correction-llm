"""
Integration tests for the end-to-end correction pipeline.

Uses explicit MockAdapter — no real models or Member 1/2 code required.
Tests cover:
    - Full pipeline: OCR → domain → correction → change detection → output
    - Field preservation: document_id, page, block, bbox, confidence, language,
      domain, original text, corrected text, correction metadata
    - Multi-page and multi-block documents
    - Domain fallback on low confidence
    - Model adapter interface compliance
"""

import pytest
from src.integration.pipeline import IntegrationPipeline
from src.integration.adapters.ocr_adapter import OCRAdapter, DomainAdapter
from src.integration.contracts import (
    CorrectedBlock,
    CorrectedDocument,
    CorrectedPage,
    DomainClassification,
    OCRBlock,
    OCRDocument,
    OCRPage,
)
from src.correction.inference import CorrectionEngine
from src.correction.model_adapter import MockAdapter, ModelAdapter


@pytest.fixture
def mock_engine():
    """CorrectionEngine with explicit MockAdapter — no model loading."""
    engine = CorrectionEngine({"model_name": "mock"}, mock_mode=True)
    engine.load_model()
    return engine


# ── Full pipeline tests ──────────────────────────────────────────────


def test_pipeline_single_block(mock_engine):
    """End-to-end: single block with 'teh' typo gets corrected."""
    pipeline = IntegrationPipeline(mock_engine)

    raw_ocr = {
        "document_id": "doc_123",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"block_id": "b1", "text": "teh cat", "bbox": [0, 0, 10, 10], "confidence": 0.9}
                ],
            }
        ],
    }
    raw_domain = {"domain": "general", "confidence": 0.95}

    result = pipeline.process_document(raw_ocr, raw_domain)

    assert result.document_id == "doc_123"
    assert len(result.pages) == 1
    assert len(result.pages[0].blocks) == 1

    block = result.pages[0].blocks[0]
    assert block.block_id == "b1"
    assert block.original_text == "teh cat"
    assert "the cat" in block.corrected_text
    assert block.domain == "general"


def test_pipeline_preserves_all_fields(mock_engine):
    """Verify every required field is preserved through the pipeline."""
    pipeline = IntegrationPipeline(mock_engine)

    raw_ocr = {
        "document_id": "doc_field_test",
        "pages": [
            {
                "page_number": 3,
                "blocks": [
                    {
                        "block_id": "blk_42",
                        "text": "teh quick brown fox",
                        "bbox": [10, 20, 300, 50],
                        "confidence": 0.87,
                    }
                ],
            }
        ],
    }
    raw_domain = {"domain": "deep_learning", "confidence": 0.92}

    result = pipeline.process_document(raw_ocr, raw_domain)

    assert result.document_id == "doc_field_test"

    page = result.pages[0]
    assert page.page_number == 3

    block = page.blocks[0]
    assert block.block_id == "blk_42"
    assert block.bbox == [10, 20, 300, 50]
    assert block.ocr_confidence == 0.87
    assert block.language != ""  # must have a language value
    assert block.domain == "deep_learning"
    assert block.original_text == "teh quick brown fox"
    assert "the quick brown fox" in block.corrected_text
    assert isinstance(block.correction_metadata, dict)
    assert isinstance(block.changes, list)


def test_pipeline_multipage(mock_engine):
    """Pipeline handles multi-page, multi-block documents."""
    pipeline = IntegrationPipeline(mock_engine)

    raw_ocr = {
        "document_id": "doc_multi",
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"block_id": "p1b1", "text": "teh first block", "bbox": [0, 0, 10, 10], "confidence": 0.95},
                    {"block_id": "p1b2", "text": "second block ok", "bbox": [0, 10, 10, 20], "confidence": 0.88},
                ],
            },
            {
                "page_number": 2,
                "blocks": [
                    {"block_id": "p2b1", "text": "teh third block", "bbox": [0, 0, 10, 10], "confidence": 0.70},
                ],
            },
        ],
    }
    raw_domain = {"domain": "medical", "confidence": 0.85}

    result = pipeline.process_document(raw_ocr, raw_domain)

    assert len(result.pages) == 2
    assert len(result.pages[0].blocks) == 2
    assert len(result.pages[1].blocks) == 1
    assert result.pages[0].blocks[0].block_id == "p1b1"
    assert result.pages[1].blocks[0].block_id == "p2b1"
    assert result.pages[1].blocks[0].domain == "medical"


# ── Adapter tests ────────────────────────────────────────────────────


def test_domain_adapter_fallback_on_low_confidence():
    """Domain adapter falls back to 'general' when confidence < 0.5."""
    result = DomainAdapter.parse_member2_domain({"domain": "legal", "confidence": 0.3})
    assert result.domain == "general"
    assert result.confidence == 0.3


def test_domain_adapter_missing_domain():
    """Domain adapter defaults to 'general' when domain key is missing."""
    result = DomainAdapter.parse_member2_domain({})
    assert result.domain == "general"


def test_domain_adapter_valid():
    """Domain adapter passes through valid classifications."""
    result = DomainAdapter.parse_member2_domain({"domain": "scientific", "confidence": 0.9})
    assert result.domain == "scientific"
    assert result.confidence == 0.9


def test_ocr_adapter_parses_valid_document():
    """OCR adapter converts raw dict to OCRDocument."""
    raw = {
        "document_id": "test_doc",
        "pages": [
            {"page_number": 1, "blocks": [{"block_id": "b1", "text": "hello", "bbox": [0, 0, 5, 5], "confidence": 0.99}]}
        ],
    }
    result = OCRAdapter.parse_member1_ocr(raw)
    assert isinstance(result, OCRDocument)
    assert result.document_id == "test_doc"
    assert result.pages[0].blocks[0].text == "hello"


# ── Model adapter interface tests ────────────────────────────────────


def test_mock_adapter_contract():
    """MockAdapter satisfies the ModelAdapter interface."""
    adapter = MockAdapter()
    assert isinstance(adapter, ModelAdapter)
    assert adapter.is_loaded() is True
    assert adapter.is_baseline is True
    assert isinstance(adapter.model_id, str)
    output = adapter.generate("test prompt")
    assert isinstance(output, str)
    assert len(output) > 0


def test_engine_with_explicit_adapter():
    """CorrectionEngine accepts an explicit adapter parameter."""
    adapter = MockAdapter()
    engine = CorrectionEngine({"model_name": "test"}, adapter=adapter)
    assert engine.adapter is adapter
    assert engine.mock_mode is True
    assert engine.is_baseline is True
