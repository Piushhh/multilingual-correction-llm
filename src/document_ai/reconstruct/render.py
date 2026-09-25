"""
Document Image Reconstruction & Correction Rendering (Task 13 / Milestone 6).

Replaces corrupted OCR text on the original document image with verified
LLM corrections, handling font matching, Devanagari script shaping, background
inpainting/color matching, and optional visual diff highlighting.
"""

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _find_system_fonts() -> Tuple[Optional[str], Optional[str]]:
    """Locate available Latin and Devanagari TrueType fonts."""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    latin_candidates = [
        Path(windir) / "Fonts" / "arial.ttf",
        Path(windir) / "Fonts" / "calibri.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    devanagari_candidates = [
        Path(windir) / "Fonts" / "mangal.ttf",
        Path(windir) / "Fonts" / "aparaj.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
        Path("/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"),
    ]

    latin_font = next((str(p) for p in latin_candidates if p.is_file()), None)
    devanagari_font = next((str(p) for p in devanagari_candidates if p.is_file()), None)
    return latin_font, devanagari_font


Sequence_or_List = Union[List[int], Tuple[int, int, int, int]]


def _estimate_background_color(image: np.ndarray, bbox: Sequence_or_List) -> Tuple[int, int, int]:
    """
    Estimate local background color around the bounding box perimeter
    to seamlessly erase corrupted text.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return (255, 255, 255)

    # Sample perimeter border (2 pixels wide around box)
    border_pixels = []
    pad = 2
    # Top border
    y_top_start = max(0, y1 - pad)
    border_pixels.append(image[y_top_start:y1, max(0, x1 - pad):min(w, x2 + pad)])
    # Bottom border
    y_bot_end = min(h, y2 + pad)
    border_pixels.append(image[y2:y_bot_end, max(0, x1 - pad):min(w, x2 + pad)])
    # Left border
    x_left_start = max(0, x1 - pad)
    border_pixels.append(image[y1:y2, x_left_start:x1])
    # Right border
    x_right_end = min(w, x2 + pad)
    border_pixels.append(image[y1:y2, x2:x_right_end])

    collected = [p.reshape(-1, 3) for p in border_pixels if p.size > 0]
    if not collected:
        # Fallback to corner pixels inside the bbox
        corners = [image[y1, x1], image[y1, x2 - 1], image[y2 - 1, x1], image[y2 - 1, x2 - 1]]
        med = np.median(corners, axis=0).astype(int)
        return (int(med[0]), int(med[1]), int(med[2]))

    stacked = np.vstack(collected)
    med = np.median(stacked, axis=0).astype(int)
    return (int(med[0]), int(med[1]), int(med[2]))


def _get_fitted_font(text: str,
                     target_width: int,
                     target_height: int,
                     latin_font_path: Optional[str],
                     devanagari_font_path: Optional[str]) -> Tuple[Any, int, int]:
    """Calculates ideal font size so text comfortably fits inside target bounding box."""
    has_devanagari = bool(_DEVANAGARI_RE.search(text))
    font_path = devanagari_font_path if (has_devanagari and devanagari_font_path) else latin_font_path

    # Start font size at ~75% of box height
    font_size = max(10, int(target_height * 0.75))
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    while font_size >= 8:
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
            break

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= target_width and text_h <= target_height:
            return font, text_w, text_h

        font_size -= 1

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    return font, bbox[2] - bbox[0], bbox[3] - bbox[1]


def render_corrections(
    original_image: Union[str, Path, np.ndarray],
    corrected_blocks: List[Any],
    out_path: Optional[Union[str, Path]] = None,
    highlight: bool = True,
) -> np.ndarray:
    """
    Inpaints and replaces erroneous OCR text with verified corrections.

    Parameters:
    - original_image: File path or numpy BGR image
    - corrected_blocks: List of CorrectedBlock objects or dictionaries
    - out_path: Optional path to save resulting rendered image
    - highlight: Whether to highlight replaced blocks with visual markers

    Returns:
    - Reconstructed numpy BGR image
    """
    if isinstance(original_image, (str, Path)):
        img = cv2.imread(str(original_image))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {original_image}")
    else:
        img = original_image.copy()

    h, w = img.shape[:2]
    latin_font_path, devanagari_font_path = _find_system_fonts()

    # Convert to PIL for unicode and font rendering
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    highlight_boxes = []

    for block in corrected_blocks:
        # Normalize block data (supporting contracts.CorrectedBlock or dict)
        if hasattr(block, "model_dump"):
            b_data = block.model_dump()
        elif hasattr(block, "dict"):
            b_data = block.dict()
        else:
            b_data = dict(block)

        orig_text = str(b_data.get("original_text", "")).strip()
        corr_text = str(b_data.get("corrected_text", "")).strip()
        bbox = b_data.get("bbox", [0, 0, 0, 0])

        if not corr_text or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        # Check if correction occurred
        is_modified = (orig_text != corr_text) or bool(b_data.get("changes"))

        if is_modified:
            # 1. Estimate background color from perimeter
            bg_bgr = _estimate_background_color(img, (x1, y1, x2, y2))
            bg_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])

            # 2. Erase / inpaint original text
            draw.rectangle([(x1, y1), (x2, y2)], fill=bg_rgb)

            # 3. Fit font and render corrected text
            font, tw, th = _get_fitted_font(
                corr_text, bw, bh, latin_font_path, devanagari_font_path
            )

            # Center text vertically and slightly indent horizontally
            tx = x1 + 2
            ty = y1 + max(0, (bh - th) // 2) - 1

            # High contrast dark text
            text_color = (15, 15, 15)
            draw.text((tx, ty), corr_text, font=font, fill=text_color)

            if highlight:
                highlight_boxes.append((x1, y1, x2, y2))

    # Convert back to OpenCV BGR
    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Apply diff highlights
    if highlight and highlight_boxes:
        overlay = result_bgr.copy()
        for x1, y1, x2, y2 in highlight_boxes:
            # Subtle green bounding rectangle for corrected regions
            cv2.rectangle(overlay, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (46, 204, 113), 2)
            # Subtle badge
            cv2.circle(overlay, (x2, y1), 4, (46, 204, 113), -1)
        # Blend overlay
        cv2.addWeighted(overlay, 0.85, result_bgr, 0.15, 0, result_bgr)

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), result_bgr)

    return result_bgr


def main():
    parser = argparse.ArgumentParser(description="Render corrected document image")
    parser.add_argument("--image", required=True, help="Path to original document image")
    parser.add_argument("--corrections", required=True, help="JSON file with list of corrected blocks")
    parser.add_argument("--out", required=True, help="Output path for reconstructed image")
    parser.add_argument("--no-highlight", action="store_true", help="Disable diff highlight boxes")
    args = parser.parse_args()

    with open(args.corrections, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("blocks", data) if isinstance(data, dict) else data
    render_corrections(args.image, blocks, out_path=args.out, highlight=not args.no_highlight)
    print(f"Reconstructed image saved to {args.out}")


if __name__ == "__main__":
    main()
