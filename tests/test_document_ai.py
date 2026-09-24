"""
Tests for Member 2 (Document AI) components:
- OCR bounding box handling and reading order
- Language detection (Unicode block analysis: en, hi, code-mixed)
- Domain classification and terminology lookup
- OCR evaluation metrics (CER, WER, exact match)
- DocumentAI Pydantic schemas and serialization
- Compatibility with Member 3's OCRDocument integration contract
"""

import pytest
from pydantic import ValidationError

from src.document_ai.ocr.bbox import polygon_to_xyxy, validate_bbox, sort_reading_order
from src.document_ai.language.detector import LanguageDetector
from src.document_ai.domain.terminology import DomainTerminologyDatabase
from src.document_ai.domain.classifier import DomainClassifier
from src.document_ai.evaluation.ocr_metrics import compute_cer, compute_wer, evaluate_ocr
from src.document_ai.schemas import DocumentAIBlock, DocumentAIPage, DocumentAIDocument
from src.integration.contracts import OCRDocument


# ── BBox & Geometry Tests ──────────────────────────────────────────

def test_polygon_to_xyxy():
    # 4-point polygon [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    polygon = [[10, 20], [100, 20], [100, 60], [10, 60]]
    bbox = polygon_to_xyxy(polygon)
    assert bbox == [10, 20, 100, 60]


def test_validate_bbox_valid():
    assert validate_bbox([0, 0, 50, 50]) == [0, 0, 50, 50]


def test_validate_bbox_invalid():
    with pytest.raises(ValueError):
        validate_bbox([50, 50, 10, 10])  # x1 > x2
    with pytest.raises(ValueError):
        validate_bbox([0, 0, 10])  # wrong length


def test_sort_reading_order():
    blocks = [
        {"bbox": [10, 120, 100, 150], "text": "Line 2"},
        {"bbox": [10, 20, 100, 50], "text": "Line 1"},
        {"bbox": [110, 25, 200, 55], "text": "Line 1 Right Column / Word"},
    ]
    sorted_blocks = sort_reading_order(blocks)
    assert sorted_blocks[0]["text"] == "Line 1"
    assert sorted_blocks[1]["text"] == "Line 1 Right Column / Word"
    assert sorted_blocks[2]["text"] == "Line 2"


# ── Language Detection Tests ────────────────────────────────────────

def test_language_detector_english():
    detector = LanguageDetector()
    assert detector.detect_language("The quick brown fox jumps over the lazy dog.") == "en"


def test_language_detector_hindi():
    detector = LanguageDetector()
    assert detector.detect_language("यह एक परीक्षण वाक्य है।") == "hi"


def test_language_detector_codemixed():
    detector = LanguageDetector()
    # Mixed Devanagari and Latin
    assert detector.detect_language("यह model बहुत fast train होता है।") == "code-mixed"


def test_language_detector_empty():
    detector = LanguageDetector()
    assert detector.detect_language("") == "unknown"
    assert detector.detect_language("123456 !!! ???") == "unknown"


# ── Domain Classification & Terminology Tests ───────────────────────

def test_domain_terminology():
    db = DomainTerminologyDatabase()
    assert db.is_domain_term("transformer", "deep_learning") is True
    assert db.is_domain_term("gradient descent", "deep_learning") is True
    assert db.is_domain_term("banana", "deep_learning") is False


def test_domain_classifier_dl():
    classifier = DomainClassifier()
    text = "We train a deep neural network using backpropagation and cross-entropy loss."
    classification = classifier.classify(text)
    assert classification["domain"] == "deep_learning"
    assert classification["confidence"] > 0.5


def test_domain_classifier_general():
    classifier = DomainClassifier()
    text = "The quick brown fox went to the supermarket to buy some groceries."
    classification = classifier.classify(text)
    assert classification["domain"] == "general"


# ── OCR Metrics Tests ───────────────────────────────────────────────

def test_ocr_metrics_cer():
    cer = compute_cer("hello", "helo")
    assert cer == pytest.approx(1.0 / 5.0)


def test_ocr_metrics_wer():
    wer = compute_wer("the quick fox", "the fast fox")
    assert wer == pytest.approx(0.3333, abs=1e-4)


def test_ocr_metrics_exact_match():
    res = evaluate_ocr("exact match test", "exact match test")
    assert res["exact_match"] == 1.0
    assert res["cer"] == 0.0
    assert res["wer"] == 0.0


# ── Schemas & Integration Contract Compatibility ────────────────────

def test_document_ai_schemas_validation():
    block = DocumentAIBlock(
        block_id="b1",
        text="Sample text",
        bbox=[0, 0, 100, 30],
        confidence=0.95,
        language="en",
    )
    assert block.block_id == "b1"

    # Invalid confidence should raise validation error
    with pytest.raises(ValidationError):
        DocumentAIBlock(
            block_id="b2",
            text="Invalid",
            bbox=[0, 0, 10, 10],
            confidence=1.5,
        )


def test_document_ai_to_integration_contract():
    """
    Ensures Member 2's DocumentAIDocument can be serialized to an
    integration dict that directly parses into Member 3's OCRDocument.
    """
    block = DocumentAIBlock(
        block_id="blk_01",
        text="Deep learning neural network",
        bbox=[10, 20, 200, 50],
        confidence=0.92,
        language="en",
    )
    page = DocumentAIPage(page_number=1, blocks=[block])
    doc = DocumentAIDocument(
        document_id="doc_mem2_01",
        pages=[page],
        domain="deep_learning",
        domain_confidence=0.88,
    )

    integration_dict = doc.to_integration_dict()
    # Parse into Member 3's OCRDocument
    ocr_doc = OCRDocument(**integration_dict)
    assert ocr_doc.document_id == "doc_mem2_01"
    assert len(ocr_doc.pages) == 1
    assert ocr_doc.pages[0].blocks[0].text == "Deep learning neural network"
    assert ocr_doc.pages[0].blocks[0].bbox == [10, 20, 200, 50]
    assert ocr_doc.pages[0].blocks[0].confidence == 0.92
