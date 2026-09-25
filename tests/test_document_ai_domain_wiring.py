"""
Tests for domain classifier wiring, fallback, and per-region language (Task 7 / Problem F).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.document_ai.domain.classifier import DomainClassifier
from src.document_ai.pipeline import process_document


def test_process_document_with_explicit_classifier(monkeypatch, tmp_path):
    img = np.full((200, 300, 3), 255, dtype=np.uint8)
    import cv2
    img_path = tmp_path / "doc.png"
    cv2.imwrite(str(img_path), img)

    # Mock extract_regions to return text regions
    monkeypatch.setattr(
        "src.document_ai.pipeline.extract_regions",
        lambda *args, **kwargs: (
            [
                {
                    "text": "Convolutional neural network training gradient descent",
                    "bbox": [10, 10, 200, 30],
                    "confidence": 0.95,
                    "words": [],
                },
                {
                    "text": "short",
                    "bbox": [10, 40, 50, 60],
                    "confidence": 0.90,
                    "words": [],
                }
            ],
            {"original_width": 300, "original_height": 200, "geometry": None}
        ),
    )

    mock_clf = MagicMock(spec=DomainClassifier)
    mock_clf.predict.return_value = {
        "domain": "deep_learning",
        "confidence": 0.88,
        "scores": {"deep_learning": 0.88, "computer_science": 0.08, "mathematics": 0.03, "general": 0.01},
    }

    result = process_document(str(img_path), page_id="page_1", domain_classifier=mock_clf)

    assert result["domain"] == "deep_learning"
    assert result["domain_confidence"] == 0.88
    assert result["domain_scores"]["deep_learning"] == 0.88
    assert len(result["regions"]) == 2

    # Region 1 length >= 8 chars -> language detected
    assert result["regions"][0]["language"] == "en"
    # Region 2 length < 8 chars -> guarded to None
    assert result["regions"][1]["language"] is None


def test_process_document_low_confidence_abstains_to_general(monkeypatch, tmp_path):
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    import cv2
    img_path = tmp_path / "low_conf.png"
    cv2.imwrite(str(img_path), img)

    monkeypatch.setattr(
        "src.document_ai.pipeline.extract_regions",
        lambda *args, **kwargs: (
            [{"text": "ambiguous text", "bbox": [5, 5, 90, 20], "confidence": 0.8, "words": []}],
            {"original_width": 100, "original_height": 100, "geometry": None}
        ),
    )

    mock_clf = MagicMock(spec=DomainClassifier)
    # Low confidence 0.32 below threshold 0.40
    mock_clf.predict.return_value = {
        "domain": "mathematics",
        "confidence": 0.32,
        "scores": {"deep_learning": 0.22, "computer_science": 0.23, "mathematics": 0.32, "general": 0.23},
    }

    result = process_document(
        str(img_path),
        page_id="page_low",
        domain_classifier=mock_clf,
        min_domain_confidence=0.40,
    )

    assert result["domain"] == "general"
    assert result["domain_confidence"] == 0.32
    assert result["domain_scores"]["mathematics"] == 0.32


def test_process_document_no_domain_flag(monkeypatch, tmp_path):
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    import cv2
    img_path = tmp_path / "no_dom.png"
    cv2.imwrite(str(img_path), img)

    monkeypatch.setattr(
        "src.document_ai.pipeline.extract_regions",
        lambda *args, **kwargs: (
            [{"text": "Sample text here", "bbox": [5, 5, 90, 20], "confidence": 0.8, "words": []}],
            {"original_width": 100, "original_height": 100, "geometry": None}
        ),
    )

    result = process_document(str(img_path), page_id="page_nodom", no_domain=True)

    assert result["domain"] is None
    assert result["domain_confidence"] is None
    assert result["domain_scores"] is None
