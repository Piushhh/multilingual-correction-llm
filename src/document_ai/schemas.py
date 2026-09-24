"""
Document AI Pydantic schemas.

Version: 1.0
These schemas define the structured representation of OCR documents produced
by Member 2, strictly compatible with Member 3's integration contracts.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DocumentAIBlock(BaseModel):
    """
    Represents an extracted text block with bounding box, confidence, and context.
    Strictly compatible with Member 3's OCRBlock contract.
    """
    block_id: str = Field(..., description="Unique block identifier within the page/document.")
    text: str = Field(..., description="Extracted OCR text.")
    bbox: List[int] = Field(..., description="Bounding box in [x1, y1, x2, y2] format.")
    confidence: float = Field(..., description="OCR confidence score in [0.0, 1.0].")
    language: str = Field("unknown", description="Detected language ('en', 'hi', 'code-mixed', or 'unknown').")
    domain: str = Field("general", description="Classified domain ('deep_learning', 'general', etc.).")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[int]) -> List[int]:
        if len(v) != 4:
            raise ValueError("Bounding box must contain exactly 4 integers [x1, y1, x2, y2]")
        x1, y1, x2, y2 = v
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            raise ValueError("Bounding box coordinates must be non-negative")
        if x2 < x1 or y2 < y1:
            raise ValueError("Invalid bounding box: x2 must be >= x1 and y2 must be >= y1")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return round(float(v), 4)


class DocumentAIPage(BaseModel):
    """
    Represents a page containing multiple OCR blocks.
    Strictly compatible with Member 3's OCRPage contract.
    """
    page_number: int = Field(..., ge=1, description="1-indexed page number.")
    blocks: List[DocumentAIBlock] = Field(default_factory=list, description="Ordered list of text blocks.")


class DocumentAIDocument(BaseModel):
    """
    Structured document understanding output.
    Strictly compatible with Member 3's OCRDocument contract.
    """
    schema_version: str = Field("1.0", description="Schema version identifier.")
    document_id: str = Field(..., description="Unique document identifier.")
    pages: List[DocumentAIPage] = Field(default_factory=list, description="Pages in the document.")

    def to_integration_dict(self) -> Dict[str, Any]:
        """
        Export dictionary cleanly consumable by Member 3's OCRDocument contract.
        """
        return {
            "document_id": self.document_id,
            "pages": [
                {
                    "page_number": p.page_number,
                    "blocks": [
                        {
                            "block_id": b.block_id,
                            "text": b.text,
                            "bbox": b.bbox,
                            "confidence": b.confidence,
                        }
                        for b in p.blocks
                    ],
                }
                for p in self.pages
            ],
        }
