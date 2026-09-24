"""
OCR extraction engine using EasyOCR baseline.
Supports English ('en') and Hindi ('hi').
Preserves raw confidence scores and reading-order bounding boxes without text alteration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from PIL import Image

from src.document_ai.ocr.bbox import polygon_to_xyxy, sort_reading_order
from src.document_ai.ocr.preprocess import preprocess_for_ocr


class OCRExtractor:
    """
    Multilingual OCR Extractor for English and Hindi document images.
    Extracts text, bounding boxes ([x1, y1, x2, y2]), and confidence scores.
    """

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        gpu: bool = False,
        reader: Optional[Any] = None,
    ):
        """
        Initialize the OCR Extractor.

        Args:
            languages: Sequence of language codes (default: ['en', 'hi']).
            gpu: Whether to use GPU acceleration for OCR.
            reader: Optional pre-initialized reader instance (useful for testing/mocking).
        """
        self.languages = list(languages or ["en", "hi"])
        self.gpu = gpu
        self._reader = reader

    @property
    def reader(self) -> Any:
        """Lazily initialize the EasyOCR reader if not already loaded."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        preprocess: bool = True,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Extract text blocks with bounding boxes and confidence from an image.

        Args:
            image_input: Path to image, numpy array, or PIL Image.
            preprocess: Whether to apply contrast enhancement before OCR.
            min_confidence: Minimum confidence threshold to keep a block.

        Returns:
            List of dicts: [
                {
                    "block_id": "block_1",
                    "text": "Extracted text string",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.92
                }, ...
            ]
        """
        if preprocess:
            processed_img = preprocess_for_ocr(image_input)
        else:
            from src.document_ai.ocr.preprocess import load_image
            processed_img = load_image(image_input)

        # EasyOCR readtext returns: [ (bbox_polygon, text, confidence), ... ]
        raw_results = self.reader.readtext(processed_img, detail=1)

        extracted_blocks: List[Dict[str, Any]] = []
        for i, item in enumerate(raw_results, start=1):
            if not item:
                continue

            # EasyOCR detail=1 format: (polygon, text, confidence)
            polygon = item[0]
            text = str(item[1]).strip()
            confidence = float(item[2]) if len(item) > 2 else 1.0

            if not text:
                continue

            if confidence < min_confidence:
                continue

            bbox = polygon_to_xyxy(polygon)

            extracted_blocks.append({
                "block_id": f"block_{i}",
                "text": text,
                "bbox": bbox,
                "confidence": round(confidence, 4),
            })

        # Sort in reading order: top-to-bottom, left-to-right
        sorted_blocks = sort_reading_order(extracted_blocks)

        # Re-number block_ids after sorting for clean sequential reading order
        for idx, block in enumerate(sorted_blocks, start=1):
            block["block_id"] = f"block_{idx}"

        return sorted_blocks
