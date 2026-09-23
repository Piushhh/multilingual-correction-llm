from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class OCRBlock(BaseModel):
    block_id: str
    text: str
    bbox: List[int]
    confidence: float
    
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
    domain: str

class CorrectedPage(BaseModel):
    page_number: int
    blocks: List[CorrectedBlock]

class CorrectedDocument(BaseModel):
    document_id: str
    pages: List[CorrectedPage]
