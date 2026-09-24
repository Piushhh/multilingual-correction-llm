"""
Integration adapters: OCRAdapter and DomainAdapter.

Member 3 owned. Minimal edits in this file per Ground Rule 2 approval:
  - parse_member2_ocr added (correct name: OCR is Member 2's responsibility)
  - parse_member1_ocr kept as a deprecated alias so existing tests still pass
"""
import warnings
from typing import Any, Dict

from src.integration.contracts import OCRDocument, DomainClassification


class OCRAdapter:
    """Adapts raw Member 2 DocumentAI outputs to the Correction pipeline contract."""

    @staticmethod
    def parse_member2_ocr(raw_data: Dict[str, Any]) -> OCRDocument:
        """
        Parse Member 2's OCR/DocumentAI structure into the pipeline OCRDocument contract.
        Expects a dict matching the OCRDocument Pydantic model fields.
        """
        return OCRDocument(**raw_data)

    @staticmethod
    def parse_member1_ocr(raw_data: Dict[str, Any]) -> OCRDocument:
        """
        DEPRECATED: Misnamed — OCR is Member 2's responsibility.
        Kept for backward compatibility. Use parse_member2_ocr instead.
        """
        warnings.warn(
            "parse_member1_ocr is deprecated and misnamed; "
            "use parse_member2_ocr instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return OCRAdapter.parse_member2_ocr(raw_data)


class DomainAdapter:
    """Adapts domain classification to the Correction pipeline standard."""

    @staticmethod
    def parse_member2_domain(raw_data: Dict[str, Any]) -> DomainClassification:
        """
        Parse Member 2's Domain Classifier into the standard contract.
        Falls back to 'general' if domain is missing or confidence < 0.5.
        """
        if not raw_data or "domain" not in raw_data:
            return DomainClassification(domain="general", confidence=1.0)

        if raw_data.get("confidence", 1.0) < 0.5:
            return DomainClassification(
                domain="general",
                confidence=raw_data.get("confidence", 1.0),
            )

        return DomainClassification(**raw_data)
