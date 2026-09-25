"""
Integration contracts — shared interface between Member 2 (document_ai)
and Member 3 (correction engine).

VERSION HISTORY:
  v1: Original minimal contract (text, bbox, confidence, language="unknown")
  v2: Added optional metadata fields from Member 2 so the correction prompt
      can use language, domain, domain_confidence, terminology_flags and
      OCR confidence. Old inputs {text, bbox, confidence} still work because
      all new fields are Optional with defaults.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class OCRBlock(BaseModel):
    block_id: str
    text: str
    bbox: List[int]
    confidence: float

    # Optional Member-2 metadata (v2 additions — all have defaults for
    # backward compatibility with existing tests and code that omit them)
    language: str = "unknown"
    language_confidence: Optional[float] = None
    domain: Optional[str] = None
    domain_confidence: Optional[float] = None
    terminology_flags: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    protected_terms: List[str] = []


class OCRPage(BaseModel):
    page_number: int
    blocks: List[OCRBlock]


class OCRDocument(BaseModel):
    document_id: str
    pages: List[OCRPage]


class DomainClassification(BaseModel):
    domain: str
    confidence: float
    alternatives: List[Dict[str, float]] = []


class CorrectedBlock(BaseModel):
    block_id: str
    original_text: str
    corrected_text: str
    changes: List[Dict[str, Any]]
    bbox: List[int]
    ocr_confidence: float
    language: str = "unknown"
    domain: str
    correction_metadata: Dict[str, Any] = {}


class CorrectedPage(BaseModel):
    page_number: int
    blocks: List[CorrectedBlock]


class CorrectedDocument(BaseModel):
    document_id: str
    pages: List[CorrectedPage]
