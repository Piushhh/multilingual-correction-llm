"""
Tests for layout-aware region extraction and reading order (Task 6 / Problem D).
"""

from src.document_ai.ocr.bbox import group_words_into_lines, order_regions_reading_order


def test_one_column_layout():
    words = [
        {"text": "Line", "bbox": [50, 100, 90, 120], "confidence": 0.95},
        {"text": "one", "bbox": [95, 100, 130, 120], "confidence": 0.95},
        {"text": "Line", "bbox": [50, 130, 90, 150], "confidence": 0.90},
        {"text": "two", "bbox": [95, 130, 130, 150], "confidence": 0.90},
    ]
    regions = group_words_into_lines(words)
    assert len(regions) == 2
    assert regions[0]["text"] == "Line one"
    assert regions[1]["text"] == "Line two"
    assert "words" in regions[0]
    assert len(regions[0]["words"]) == 2


def test_two_column_layout_never_merges_columns():
    """
    Two columns side-by-side at the same vertical positions:
    Col 1 (left): x in [50, 250]
    Col 2 (right): x in [400, 600]
    Words at y=100 and y=130 in both columns must NOT merge across the gutter.
    """
    words = [
        # Left column line 1 (y=100)
        {"text": "Left", "bbox": [50, 100, 100, 120], "confidence": 0.9},
        {"text": "col", "bbox": [110, 100, 150, 120], "confidence": 0.9},
        {"text": "one", "bbox": [160, 100, 200, 120], "confidence": 0.9},
        # Right column line 1 (y=100)
        {"text": "Right", "bbox": [400, 100, 460, 120], "confidence": 0.9},
        {"text": "col", "bbox": [470, 100, 510, 120], "confidence": 0.9},
        {"text": "one", "bbox": [520, 100, 570, 120], "confidence": 0.9},
        # Left column line 2 (y=130)
        {"text": "Left", "bbox": [50, 130, 100, 150], "confidence": 0.9},
        {"text": "line", "bbox": [110, 130, 150, 150], "confidence": 0.9},
        {"text": "two", "bbox": [160, 130, 200, 150], "confidence": 0.9},
        # Right column line 2 (y=130)
        {"text": "Right", "bbox": [400, 130, 460, 150], "confidence": 0.9},
        {"text": "line", "bbox": [470, 130, 510, 150], "confidence": 0.9},
        {"text": "two", "bbox": [520, 130, 570, 150], "confidence": 0.9},
    ]

    regions = group_words_into_lines(words)

    # Must produce 4 distinct regions, not 2 merged lines spanning across both columns
    assert len(regions) == 4, f"Expected 4 regions, got {len(regions)}"

    # Reading order check: Left column lines first, then Right column lines
    texts = [r["text"] for r in regions]
    assert texts[0] == "Left col one"
    assert texts[1] == "Left line two"
    assert texts[2] == "Right col one"
    assert texts[3] == "Right line two"

    # Verify no region spans across the gutter (x_max < 300 for left, x_min > 350 for right)
    for r in regions[:2]:
        assert r["bbox"][2] <= 250, f"Left column region crossed into right column: {r['bbox']}"
    for r in regions[2:]:
        assert r["bbox"][0] >= 350, f"Right column region crossed into left column: {r['bbox']}"


def test_mixed_english_hindi_line():
    words = [
        {"text": "यह", "bbox": [40, 50, 70, 75], "confidence": 0.92},
        {"text": "transformer", "bbox": [80, 50, 200, 75], "confidence": 0.95},
        {"text": "model", "bbox": [210, 50, 270, 75], "confidence": 0.94},
        {"text": "है", "bbox": [280, 50, 310, 75], "confidence": 0.91},
    ]
    regions = group_words_into_lines(words)
    assert len(regions) == 1
    assert regions[0]["text"] == "यह transformer model है"
    assert len(regions[0]["words"]) == 4
