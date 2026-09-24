"""
Regression tests for deskew (problem A).

A text page rotated by a known angle must come back to within 0.5° of
upright after deskew. Residual skew is measured with a projection-profile
search that does not call the production estimator, so a wrong-sign
minAreaRect correction cannot hide itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.document_ai.ocr.preprocess import deskew, estimate_skew_angle, to_grayscale


def _find_latin_font() -> str | None:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        Path(windir) / "Fonts" / "arial.ttf",
        Path(windir) / "Fonts" / "calibri.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _render_text_page(width: int = 900, height: int = 1200) -> np.ndarray:
    """Synthetic upright page of horizontal Latin lines (white background)."""
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_path = _find_latin_font()
    font = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()

    lines = [
        "Gradient descent updates the weights of the network.",
        "Attention is a core mechanism in transformer models.",
        "Backpropagation computes gradients for every layer.",
        "Convolutional networks extract local spatial features.",
        "Dropout is a regularisation method used in training.",
        "Batch normalisation stabilises the learning dynamics.",
        "Cross entropy is a common classification objective.",
        "Stochastic gradient descent uses mini-batch estimates.",
        "Residual connections ease optimisation of deep stacks.",
        "Layer normalisation is common in transformer blocks.",
        "Adam combines momentum with adaptive step sizes.",
        "Early stopping is a simple regularisation heuristic.",
    ]
    y = 80
    for line in lines:
        draw.text((60, y), line, fill=(0, 0, 0), font=font)
        y += 70

    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return to_grayscale(bgr)


def _rotate_same_size(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around centre; positive angle is counter-clockwise (OpenCV)."""
    height, width = gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _projection_variance(gray: np.ndarray, angle_deg: float) -> float:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rotated = _rotate_same_size(binary, angle_deg)
    projection = np.sum(rotated > 0, axis=1, dtype=np.float64)
    return float(np.var(projection))


def independent_residual_skew_degrees(gray: np.ndarray) -> float:
    """
    Angle (CCW degrees) that maximises horizontal projection variance.

    This is the residual rotation still present in `gray`. Independent of
    `estimate_skew_angle` so a wrong-sign production estimator cannot
    cancel itself in the assertion.
    """
    best_angle = 0.0
    best_score = -1.0
    for angle in np.arange(-12.0, 12.0 + 1e-9, 0.25):
        score = _projection_variance(gray, float(angle))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


@pytest.fixture(scope="module")
def upright_page() -> np.ndarray:
    return _render_text_page()


@pytest.mark.parametrize("applied_rotation", [-10.0, -3.0, 0.0, 3.0, 10.0])
def test_deskew_residual_skew_under_half_degree(upright_page, applied_rotation):
    skewed = _rotate_same_size(upright_page, applied_rotation)
    deskewed = deskew(skewed)
    residual = independent_residual_skew_degrees(deskewed)
    assert abs(residual) < 0.5, (
        f"applied rotation {applied_rotation}° left residual {residual:.2f}° "
        f"(estimate_skew_angle returned {estimate_skew_angle(skewed):.2f}°)"
    )


def test_deskew_does_not_double_positive_skew(upright_page):
    """The original minAreaRect bug doubled +3° into ~+6°."""
    skewed = _rotate_same_size(upright_page, 3.0)
    deskewed = deskew(skewed)
    residual = independent_residual_skew_degrees(deskewed)
    assert abs(residual) < 1.5, f"deskew doubled or worsened +3° skew (residual={residual:.2f})"
    assert abs(residual) < abs(3.0) - 0.5
