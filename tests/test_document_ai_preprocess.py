"""Tests for PreprocessConfig, quality gating, and default pipeline (problem B)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.document_ai.ocr.preprocess import (
    DEFAULT_PREPROCESS_CONFIG,
    PreprocessConfig,
    page_quality_stats,
    preprocess_array,
    preprocess_image,
    sauvola_binarize,
)


def _latin_font():
    import os
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for p in (
        Path(windir) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if p.is_file():
            return str(p)
    return None


def _clean_page_bgr(width=640, height=800):
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_path = _latin_font()
    font = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()
    y = 60
    for i in range(8):
        draw.text((40, y), f"Clean high-contrast line number {i}", fill=(0, 0, 0), font=font)
        y += 50
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def test_default_config_is_light_denoise_only():
    cfg = DEFAULT_PREPROCESS_CONFIG
    assert cfg.denoise is True
    assert cfg.contrast is False
    assert cfg.deskew is False
    assert cfg.binarize == "none"
    assert cfg.auto_quality is True


def test_default_preprocess_keeps_grayscale_not_binary():
    page = _clean_page_bgr()
    result = preprocess_array(page)
    unique = len(np.unique(result.image))
    assert unique > 2, f"default preprocess binarized a clean page ({unique} unique values)"
    assert result.applied["binarize"] == "none"
    assert result.applied["deskew"] is False


def test_quality_stats_flag_low_contrast():
    clean = cv2.cvtColor(_clean_page_bgr(), cv2.COLOR_BGR2GRAY)
    faded = (clean.astype(np.float32) * 0.15 + 180).clip(0, 255).astype(np.uint8)
    clean_stats = page_quality_stats(clean)
    faded_stats = page_quality_stats(faded)
    assert clean_stats["low_contrast"] is False
    assert faded_stats["low_contrast"] is True


def test_auto_quality_enables_contrast_on_faded_page():
    clean = _clean_page_bgr()
    faded = (clean.astype(np.float32) * 0.12 + 190).clip(0, 255).astype(np.uint8)
    result = preprocess_array(faded, config=PreprocessConfig(resize=False, denoise=False))
    assert result.applied["contrast"] is True


def test_legacy_do_binarize_still_works(tmp_path):
    page = _clean_page_bgr()
    path = tmp_path / "page.png"
    cv2.imwrite(str(path), page)
    binary = preprocess_image(str(path), do_denoise=False, do_contrast=False, do_deskew=False, do_binarize=True)
    assert set(np.unique(binary)).issubset({0, 255})


def test_sauvola_returns_binary():
    gray = cv2.cvtColor(_clean_page_bgr(), cv2.COLOR_BGR2GRAY)
    out = sauvola_binarize(gray)
    assert set(np.unique(out)).issubset({0, 255})


def test_backward_compatible_signature(tmp_path):
    page = _clean_page_bgr()
    path = tmp_path / "page.png"
    cv2.imwrite(str(path), page)
    out = preprocess_image(str(path))
    assert out.ndim == 2
    assert out.dtype == np.uint8
