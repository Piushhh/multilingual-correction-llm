"""
Document AI Pipeline.
Coordinates: Image -> Preprocessing -> OCR -> Language Detection -> Domain Classification -> DocumentAIDocument.
Produces structured document understanding output compatible with Member 3's integration contracts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import numpy as np
from PIL import Image

from src.document_ai.domain.classifier import DomainClassifier
from src.document_ai.language.detector import LanguageDetector
from src.document_ai.ocr.extract import OCRExtractor
from src.document_ai.schemas import DocumentAIBlock, DocumentAIDocument, DocumentAIPage


class DocumentAIPipeline:
    """
    End-to-end Document AI processing pipeline.
    Transforms raw document images into structured, enriched DocumentAIDocument models.
    """

    def __init__(
        self,
        ocr_extractor: Optional[OCRExtractor] = None,
        language_detector: Optional[LanguageDetector] = None,
        domain_classifier: Optional[DomainClassifier] = None,
    ):
        """
        Initialize the Document AI Pipeline.

        Args:
            ocr_extractor: OCRExtractor instance (lazily initialized if None).
            language_detector: LanguageDetector instance.
            domain_classifier: DomainClassifier instance.
        """
        self.ocr_extractor = ocr_extractor or OCRExtractor()
        self.language_detector = language_detector or LanguageDetector()
        self.domain_classifier = domain_classifier or DomainClassifier()

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        document_id: Optional[str] = None,
        page_number: int = 1,
        preprocess: bool = True,
    ) -> DocumentAIDocument:
        """
        Process a single document page image.

        Args:
            image_input: File path, numpy image, or PIL Image.
            document_id: Optional unique document identifier.
            page_number: Page number (1-indexed).
            preprocess: Whether to apply contrast enhancement.

        Returns:
            Validated DocumentAIDocument instance.
        """
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"

        # 1. OCR Extraction (bounding boxes + raw text + confidence)
        raw_blocks = self.ocr_extractor.extract(image_input, preprocess=preprocess)

        processed_blocks: List[DocumentAIBlock] = []
        for blk in raw_blocks:
            text = blk["text"]

            # 2. Language Detection
            lang_res = self.language_detector.detect(text)
            detected_lang = lang_res.get("language", "unknown")

            # 3. Domain Classification
            domain_res = self.domain_classifier.classify(text)
            detected_domain = domain_res.get("domain", "general")

            processed_blocks.append(
                DocumentAIBlock(
                    block_id=blk["block_id"],
                    text=text,
                    bbox=blk["bbox"],
                    confidence=blk["confidence"],
                    language=detected_lang,
                    domain=detected_domain,
                )
            )

        page = DocumentAIPage(
            page_number=page_number,
            blocks=processed_blocks,
        )

        return DocumentAIDocument(
            schema_version="1.0",
            document_id=doc_id,
            pages=[page],
        )

    def process_document(
        self,
        pages: List[Union[str, Path, np.ndarray, Image.Image]],
        document_id: Optional[str] = None,
        preprocess: bool = True,
    ) -> DocumentAIDocument:
        """
        Process multi-page documents.
        """
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        doc_pages: List[DocumentAIPage] = []

        for idx, page_img in enumerate(pages, start=1):
            single_page_doc = self.process_image(
                page_img,
                document_id=doc_id,
                page_number=idx,
                preprocess=preprocess,
            )
            doc_pages.extend(single_page_doc.pages)

        return DocumentAIDocument(
            schema_version="1.0",
            document_id=doc_id,
            pages=doc_pages,
        )
