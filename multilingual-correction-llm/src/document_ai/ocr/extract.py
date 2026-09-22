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

from pathlib import Path

import pytesseract
from pytesseract import Output

from src.document_ai.ocr.preprocess import preprocess_image
from src.document_ai.ocr.bbox import to_xyxy, group_words_into_lines


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
            }
        )

    return words


def extract_regions(
    image_path,
    languages=DEFAULT_LANGUAGES,
    min_confidence=0,
    preprocess=True,
):
    """
    Full extraction pipeline for one image: preprocess -> OCR -> group words
    into line-level regions.

    Returns a list of region dicts matching the team's agreed interface
    shape: [{"text": ..., "bbox": [...], "confidence": ...}, ...]
    (language/domain are added later by the pipeline, not here -- this
    module's only job is turning pixels into text + geometry).
    """
    image = preprocess_image(image_path) if preprocess else _load_raw(image_path)

    words = run_tesseract(image, languages=languages, min_confidence=min_confidence)

    return group_words_into_lines(words)


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
