"""
src.integration — Integration pipeline package (Member 3).

Provides end-to-end pipeline connecting OCR output (Member 2),
domain classification (Member 2), and correction (Member 3).

Public exports:
    - IntegrationPipeline: End-to-end document processing
    - Pydantic contracts: OCRDocument, OCRPage, OCRBlock,
      DomainClassification, CorrectedDocument, CorrectedPage, CorrectedBlock
"""

from src.integration.contracts import (
    OCRBlock,
    OCRPage,
    OCRDocument,
    DomainClassification,
    CorrectedBlock,
    CorrectedPage,
    CorrectedDocument,
)
from src.integration.pipeline import IntegrationPipeline

__all__ = [
    "OCRBlock",
    "OCRPage",
    "OCRDocument",
    "DomainClassification",
    "CorrectedBlock",
    "CorrectedPage",
    "CorrectedDocument",
    "IntegrationPipeline",
]
