"""
Tests for Document Image Reconstruction (Task 13).
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.document_ai.reconstruct.render import render_corrections
from src.integration.contracts import CorrectedBlock


def test_render_corrections_numpy_array(tmp_path):
    # Create clean synthetic image (white canvas)
    img = np.full((200, 400, 3), 255, dtype=np.uint8)

    # Put some initial black text
    cv2.putText(img, "Initial err text", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    blocks = [
        {
            "block_id": "b1",
            "original_text": "Initial err text",
            "corrected_text": "Corrected clean text",
            "bbox": [15, 30, 250, 60],
            "changes": [{"type": "spelling"}],
        }
    ]

    out_file = tmp_path / "recon_test.png"
    result = render_corrections(img, blocks, out_path=out_file, highlight=True)

    assert result.shape == (200, 400, 3)
    assert out_file.exists()


def test_render_corrections_with_contract_model(tmp_path):
    img = np.full((150, 300, 3), 255, dtype=np.uint8)

    block = CorrectedBlock(
        block_id="block_001",
        original_text="The transfrmer model",
        corrected_text="The transformer model",
        changes=[{"from": "transfrmer", "to": "transformer"}],
        bbox=[20, 20, 260, 50],
        ocr_confidence=0.85,
        language="en",
        domain="deep_learning",
    )

    out_file = tmp_path / "contract_recon.png"
    result = render_corrections(img, [block], out_path=out_file, highlight=False)

    assert result.shape == (150, 300, 3)
    assert out_file.exists()


def test_render_corrections_devanagari(tmp_path):
    img = np.full((150, 300, 3), 255, dtype=np.uint8)

    blocks = [
        {
            "block_id": "b_hi",
            "original_text": "गलत पाठ",
            "corrected_text": "सही पाठ",
            "bbox": [20, 20, 200, 60],
            "changes": [{"type": "correction"}],
        }
    ]

    result = render_corrections(img, blocks, highlight=True)
    assert result.shape == (150, 300, 3)
