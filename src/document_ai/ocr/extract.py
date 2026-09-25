"""
OCR extraction: image -> text regions with bounding boxes and confidence.

Member 2, Responsibility 2: "Extract text, bounding boxes, confidence
values, and region metadata from images."

Uses Tesseract (via pytesseract) since it's free, runs locally, and has
solid multi-language support including Hindi (`hin`) out of the box --
important since this is a requirement, not a nice-to-have, per the team's
English+Hindi scope.

Requires the Tesseract binary itself to be installed separately from the
Python package (pytesseract is just a wrapper):
  - Ubuntu/Debian: apt-get install tesseract-ocr tesseract-ocr-hin
  - macOS:         brew install tesseract tesseract-lang
  - Windows:        https://github.com/UB-Mannheim/tesseract/wiki
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytesseract
from pytesseract import Output

from src.document_ai.ocr.preprocess import (
    GeometryTransform,
    PreprocessConfig,
    preprocess_image,
    preprocess_with_geometry,
)
from src.document_ai.ocr.bbox import to_xyxy, group_words_into_lines


def configure_tesseract(tesseract_cmd: str = None) -> str:
    """
    Configure the pytesseract binary path from parameter, env var, or standard locations.
    Returns the resolved path, or None if not found.
    """
    cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD")

    if cmd and os.path.isfile(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        return cmd
    return getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")


def is_tesseract_available() -> bool:
    """Check if Tesseract binary is accessible and executable."""
    cmd = configure_tesseract()
    if cmd and os.path.isfile(cmd):
        return True
    return shutil.which(cmd) is not None


# Run auto-configuration on module load
configure_tesseract()

# Tesseract language codes. "eng+hin" runs both models and lets Tesseract
# pick per-token -- this is what makes code-mixed En-Hi text usable without
# a separate pass.
DEFAULT_LANGUAGES = "eng+hin"


def run_tesseract(image, languages=DEFAULT_LANGUAGES, min_confidence=0):
    """
    Run Tesseract on a preprocessed (grayscale/binary) image and return
    raw word-level detections as a list of dicts:
    {text, bbox, confidence}.

    Tesseract's confidence is on a 0-100 scale; -1 means "no confidence
    value" (e.g. for non-text regions) and those are dropped.
    """
    data = pytesseract.image_to_data(
        image,
        lang=languages,
        output_type=Output.DICT,
    )

    words = []
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        text = data["text"][i].strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1.0

        if confidence < 0 or confidence < min_confidence:
            continue

        box = to_xyxy(
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        words.append(
            {
                "text": text,
                "bbox": box,
                # Store as 0-1 to match the interface's `confidence` field,
                # which is documented as 0-1 in the team's example output.
                "confidence": round(confidence / 100.0, 4),
                "block_num": data.get("block_num", [0] * n_boxes)[i],
                "par_num": data.get("par_num", [0] * n_boxes)[i],
                "line_num": data.get("line_num", [0] * n_boxes)[i],
            }
        )

    return words


def extract_regions(
    image_path,
    languages=DEFAULT_LANGUAGES,
    min_confidence=0,
    preprocess=True,
    config: Optional[PreprocessConfig] = None,
    return_metadata: bool = False,
):
    """
    Full extraction pipeline for one image: preprocess -> OCR -> map boxes to
    original image space -> group words into line-level regions.

    Returns a list of region dicts matching the team's agreed interface
    shape: [{"text": ..., "bbox": [...], "confidence": ..., "words": [...]}, ...]
    All bbox values are mapped back to original image coordinates (problem C).

    If return_metadata is True, returns (regions, metadata_dict).
    """
    if preprocess:
        prep_res = preprocess_with_geometry(image_path, config=config)
        image = prep_res.image
        geometry = prep_res.geometry
    else:
        image = _load_raw(image_path)
        h, w = image.shape[:2]
        geometry = GeometryTransform(
            original_width=w,
            original_height=h,
            scale=1.0,
            rotation_deg=0.0,
            processed_width=w,
            processed_height=h,
        )

    metadata = {
        "original_width": geometry.original_width,
        "original_height": geometry.original_height,
        "geometry": geometry,
    }

    if not is_tesseract_available():
        import logging
        logging.getLogger(__name__).warning("Tesseract is not available on this system.")
        return ([], metadata) if return_metadata else []

    words = run_tesseract(image, languages=languages, min_confidence=min_confidence)

    # Map word bboxes back to original image space
    mapped_words = []
    for w in words:
        mapped_box = geometry.map_bbox_to_original(w["bbox"])
        mw = dict(w)
        mw["processed_bbox"] = w["bbox"]
        mw["bbox"] = mapped_box
        mapped_words.append(mw)

    regions = group_words_into_lines(mapped_words)

    if return_metadata:
        return regions, metadata
    return regions


def _load_raw(image_path):
    import cv2

    image = cv2.imread(str(Path(image_path)))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_full_text(regions):
    """Concatenate region texts (in reading order) into a single string --
    convenient for feeding the domain classifier and terminology checker,
    which both operate on page-level text rather than per-region."""
    return "\n".join(r["text"] for r in regions)
