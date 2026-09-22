"""
Unit tests for Member 2's components that don't require Tesseract/OpenCV
runtime behavior (those are covered by manual/integration testing per
src/document_ai/README.md, since they depend on the system Tesseract
install). These cover the pure-Python logic: bbox math, language detection,
terminology matching, and evaluation metrics.

Run with: pytest tests/test_document_ai.py
"""

from src.document_ai.ocr.bbox import (
    intersection_over_union,
    vertical_overlap_ratio,
    group_words_into_lines,
    union_box,
)
from src.document_ai.language.detector import detect_language, script_composition
from src.document_ai.domain.terminology import TerminologyDatabase
from src.document_ai.evaluation.metrics import (
    character_error_rate,
    word_error_rate,
    evaluate_ocr,
)


# ---------- bbox.py ----------

def test_iou_identical_boxes():
    box = [0, 0, 10, 10]
    assert intersection_over_union(box, box) == 1.0


def test_iou_disjoint_boxes():
    assert intersection_over_union([0, 0, 5, 5], [100, 100, 110, 110]) == 0.0


def test_union_box():
    assert union_box([0, 0, 5, 5], [3, 3, 10, 10]) == [0, 0, 10, 10]


def test_vertical_overlap_same_line():
    word_a = [0, 100, 50, 120]
    word_b = [60, 102, 110, 122]
    assert vertical_overlap_ratio(word_a, word_b) > 0.8


def test_group_words_into_lines_two_lines():
    words = [
        {"text": "Hello", "bbox": [0, 0, 50, 20], "confidence": 0.9},
        {"text": "world", "bbox": [55, 2, 100, 22], "confidence": 0.9},
        {"text": "Second", "bbox": [0, 40, 60, 60], "confidence": 0.9},
        {"text": "line", "bbox": [65, 42, 100, 62], "confidence": 0.9},
    ]
    lines = group_words_into_lines(words)
    assert len(lines) == 2
    assert lines[0]["text"] == "Hello world"
    assert lines[1]["text"] == "Second line"


# ---------- language/detector.py ----------

def test_script_composition_pure_english():
    comp = script_composition("The quick brown fox jumps over the lazy dog")
    assert comp["latin"] > 0
    assert comp["devanagari"] == 0


def test_detect_language_pure_hindi():
    result = detect_language("यह एक हिन्दी वाक्य है जो देवनागरी लिपि में लिखा गया है")
    assert result["language"] == "hi"


def test_detect_language_code_mixed():
    text = "यह transformer model बहुत अच्छा काम करता है और attention mechanism use करता है"
    result = detect_language(text)
    assert result["language"] == "code_mixed"


def test_detect_language_empty_text():
    result = detect_language("")
    assert result["language"] == "unknown"
    assert result["confidence"] == 0.0


# ---------- domain/terminology.py ----------

def test_terminology_exact_match_not_flagged_as_error():
    db = TerminologyDatabase({"deep_learning": ["attention", "gradient"]})
    findings = db.check_text("The model uses attention and gradient descent.")
    # Exact matches are skipped by default (exact_match_skip=True)
    assert all(f["status"] != "possible_ocr_error" for f in findings if f["word"].lower() in ("attention", "gradient"))


def test_terminology_detects_ocr_error():
    db = TerminologyDatabase({"deep_learning": ["attention"]})
    findings = db.check_text("The model use atention.")
    flagged_words = [f["word"] for f in findings]
    assert "atention" in flagged_words
    finding = next(f for f in findings if f["word"] == "atention")
    assert finding["status"] == "possible_ocr_error"
    assert finding["closest_term"] == "attention"


def test_terminology_no_match_for_unrelated_word():
    db = TerminologyDatabase({"deep_learning": ["attention"]})
    findings = db.check_text("The weather is nice today.")
    assert findings == []


# ---------- evaluation/metrics.py ----------

def test_character_error_rate_identical():
    assert character_error_rate("hello", "hello") == 0.0


def test_character_error_rate_one_substitution():
    # "hallo" vs "hello": 1 edit / 5 chars
    assert character_error_rate("hallo", "hello") == 0.2


def test_word_error_rate_one_deletion():
    # hyp is missing "brown": 1 edit / 4 ref words
    assert word_error_rate("the quick fox jumps", "the quick brown fox jumps") == 0.2


def test_evaluate_ocr_aggregates_mean():
    pairs = [("hello", "hello"), ("hallo", "hello")]
    result = evaluate_ocr(pairs)
    assert result["n"] == 2
    assert result["mean_cer"] == 0.1  # (0.0 + 0.2) / 2
