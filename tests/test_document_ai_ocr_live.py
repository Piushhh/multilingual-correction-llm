"""
Live Tesseract OCR Tests (Task 2).

These tests require a live system installation of Tesseract OCR with both
English ('eng') and Hindi ('hin') trained data. In local environments lacking
Tesseract, they are safely skipped. In GitHub Actions CI (where Tesseract is
installed via apt-get), both tests execute and verify live extraction.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from src.document_ai.ocr.extract import extract_regions, is_tesseract_available
from scripts.generate_ocr_eval_set import find_fonts


@pytest.mark.ocr
@pytest.mark.skipif(
    not is_tesseract_available(),
    reason="Tesseract OCR binary not installed on host — runs in CI",
)
def test_live_tesseract_english_extraction(tmp_path: Path):
    """Verifies live Tesseract extraction on rendered English technical text."""
    latin_font, _ = find_fonts()
    font = ImageFont.truetype(latin_font, 28) if latin_font else ImageFont.load_default()

    img = Image.new("RGB", (650, 150), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 50), "Convolutional Neural Network and Transformer", fill=(0, 0, 0), font=font)

    test_img_path = tmp_path / "live_en_test.png"
    img.save(test_img_path)

    regions, meta = extract_regions(str(test_img_path), languages="eng", return_metadata=True)

    assert len(regions) > 0
    extracted_text = " ".join(r["text"] for r in regions).lower()
    assert "network" in extracted_text or "neural" in extracted_text or "transformer" in extracted_text
    assert meta["original_width"] == 650
    assert meta["original_height"] == 150


@pytest.mark.ocr
@pytest.mark.skipif(
    not is_tesseract_available(),
    reason="Tesseract OCR binary not installed on host — runs in CI",
)
def test_live_tesseract_hindi_extraction(tmp_path: Path):
    """Verifies live Tesseract extraction on rendered Hindi Devanagari text."""
    _, devanagari_font = find_fonts()
    if not devanagari_font:
        pytest.skip("Devanagari font not found on host system")

    font = ImageFont.truetype(devanagari_font, 28)
    img = Image.new("RGB", (650, 150), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 50), "गहन शिक्षण और कंप्यूटर विज्ञान", fill=(0, 0, 0), font=font)

    test_img_path = tmp_path / "live_hi_test.png"
    img.save(test_img_path)

    regions, meta = extract_regions(str(test_img_path), languages="hin", return_metadata=True)

    assert len(regions) > 0
    extracted_text = " ".join(r["text"] for r in regions)
    # Verify Devanagari characters extracted
    has_devanagari = any("\u0900" <= c <= "\u097f" for c in extracted_text)
    assert has_devanagari
