"""
Generates 3 before/after visual demonstration figures for image reconstruction (Task 13).
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.document_ai.reconstruct.render import render_corrections, _find_system_fonts

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

latin_font, devanagari_font = _find_system_fonts()

def create_sample_page(title: str, lines: list, lang: str):
    w, h = 700, 320
    img = Image.new("RGB", (w, h), (250, 250, 248))
    draw = ImageDraw.Draw(img)

    f_title = ImageFont.truetype(latin_font, 22) if latin_font else ImageFont.load_default()
    font_path = devanagari_font if lang in ("hi", "code_mixed") and devanagari_font else latin_font
    f_body = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()

    draw.text((40, 30), title, font=f_title, fill=(30, 30, 30))

    y = 80
    line_boxes = []
    for line in lines:
        draw.text((40, y), line, font=f_body, fill=(40, 40, 40))
        bbox = draw.textbbox((40, y), line, font=f_body)
        line_boxes.append([bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3])
        y += 45

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), line_boxes


def make_comparison_figure(original_bgr, reconstructed_bgr, out_path, title):
    # Place side-by-side with headers
    h, w = original_bgr.shape[:2]
    header_h = 50
    canvas = np.full((h + header_h, w * 2 + 30, 3), 245, dtype=np.uint8)

    # Copy images
    canvas[header_h:header_h + h, 10:10 + w] = original_bgr
    canvas[header_h:header_h + h, 20 + w:20 + w * 2] = reconstructed_bgr

    # Draw header text
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)
    f_hdr = ImageFont.truetype(latin_font, 20) if latin_font else ImageFont.load_default()

    draw.text((20, 15), f"Corrupted OCR Input ({title})", font=f_hdr, fill=(180, 40, 40))
    draw.text((30 + w, 15), f"Reconstructed Output ({title})", font=f_hdr, fill=(40, 140, 40))

    final_bgr = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), final_bgr)
    print(f"Saved {out_path}")


def generate_demos():
    # 1. English Demo
    en_err_lines = [
        "The transfrmer model uses multi-head atention.",
        "Backpr0pagation computes gradients with autograd.",
        "Regulrization prevents sever overfitting.",
    ]
    en_corr_lines = [
        "The transformer model uses multi-head attention.",
        "Backpropagation computes gradients with autograd.",
        "Regularization prevents severe overfitting.",
    ]
    en_img, en_boxes = create_sample_page("English Deep Learning Notes", en_err_lines, "en")
    en_blocks = [
        {
            "block_id": f"b_{i}",
            "original_text": en_err_lines[i],
            "corrected_text": en_corr_lines[i],
            "bbox": en_boxes[i],
            "changes": [{"type": "spelling_fix"}],
        }
        for i in range(len(en_err_lines))
    ]
    en_recon = render_corrections(en_img, en_blocks, highlight=True)
    make_comparison_figure(en_img, en_recon, OUT_DIR / "reconstruct_demo_en.png", "English")

    # 2. Hindi Demo
    hi_err_lines = [
        "डेटा संसचनाएं और एल्रगोरिदम समस्याओं का हल हैं।",
        "अवकलन किसी फलन के परिवत्तन की दर को दर्शाता है।",
        "रैखिक बीजगणित सदिश अंतरिछ का अध्ययन करता है।",
    ]
    hi_corr_lines = [
        "डेटा संरचनाएं और एल्गोरिदम समस्याओं का हल हैं।",
        "अवकलन किसी फलन के परिवर्तन की दर को दर्शाता है।",
        "रैखिक बीजगणित सदिश अंतरिक्ष का अध्ययन करता है।",
    ]
    hi_img, hi_boxes = create_sample_page("Hindi Technical Science", hi_err_lines, "hi")
    hi_blocks = [
        {
            "block_id": f"b_{i}",
            "original_text": hi_err_lines[i],
            "corrected_text": hi_corr_lines[i],
            "bbox": hi_boxes[i],
            "changes": [{"type": "devanagari_fix"}],
        }
        for i in range(len(hi_err_lines))
    ]
    hi_recon = render_corrections(hi_img, hi_blocks, highlight=True)
    make_comparison_figure(hi_img, hi_recon, OUT_DIR / "reconstruct_demo_hi.png", "Hindi")

    # 3. Code-Mixed Demo
    cm_err_lines = [
        "यह transfrmer model multi-head atention use करता है।",
        "Deep learning में backpr0pagation से weights optimize होते हैं।",
        "Overfitting से बचने के लिए regulrization बहुत जरूरी है।",
    ]
    cm_corr_lines = [
        "यह transformer model multi-head attention use करता है।",
        "Deep learning में backpropagation से weights optimize होते हैं।",
        "Overfitting से बचने के लिए regularization बहुत जरूरी है।",
    ]
    cm_img, cm_boxes = create_sample_page("Code-Mixed Technical AI", cm_err_lines, "code_mixed")
    cm_blocks = [
        {
            "block_id": f"b_{i}",
            "original_text": cm_err_lines[i],
            "corrected_text": cm_corr_lines[i],
            "bbox": cm_boxes[i],
            "changes": [{"type": "codemix_fix"}],
        }
        for i in range(len(cm_err_lines))
    ]
    cm_recon = render_corrections(cm_img, cm_blocks, highlight=True)
    make_comparison_figure(cm_img, cm_recon, OUT_DIR / "reconstruct_demo_cm.png", "Code-Mixed")


if __name__ == "__main__":
    generate_demos()
