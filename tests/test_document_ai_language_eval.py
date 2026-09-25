"""
Tests for Language Detection Evaluation (Task 12).
"""

from scripts.eval_language_detection import LABELED_CORPUS
from src.document_ai.evaluation.metrics import evaluate_language_detection
from src.document_ai.language.detector import detect_language


def test_language_detection_corpus_size():
    assert len(LABELED_CORPUS) >= 150
    classes = {lang for _, lang in LABELED_CORPUS}
    assert classes == {"en", "hi", "code_mixed"}


def test_language_detection_accuracy():
    results = evaluate_language_detection(LABELED_CORPUS)
    assert results["n"] >= 150
    assert results["accuracy"] >= 0.85
