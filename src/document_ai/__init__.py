"""
Member 2: Document AI Package.
Contains OCR extraction, Language Detection, Domain Classification,
Terminology Database, and Document Understanding Pipeline.
"""

from src.document_ai.domain import DomainClassifier, DomainTerminologyDatabase
from src.document_ai.language import LanguageDetector, detect_language
from src.document_ai.ocr import OCRExtractor, polygon_to_xyxy, preprocess_for_ocr, validate_bbox
from src.document_ai.pipeline import DocumentAIPipeline
from src.document_ai.schemas import DocumentAIBlock, DocumentAIDocument, DocumentAIPage

__all__ = [
    "DocumentAIPipeline",
    "DocumentAIDocument",
    "DocumentAIPage",
    "DocumentAIBlock",
    "OCRExtractor",
    "LanguageDetector",
    "detect_language",
    "DomainClassifier",
    "DomainTerminologyDatabase",
    "preprocess_for_ocr",
    "polygon_to_xyxy",
    "validate_bbox",
]
