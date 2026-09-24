"""
Integration pipeline: Member 3 owned.

Minimal approved Stage 5 edit: replace hardcoded language="unknown" with the
language coming from the OCR block, and forward domain, domain_confidence,
terminology_flags, and ocr_confidence into the correction request so the
correction engine has the full Member 2 metadata without re-discovering it.
"""
from typing import Any, Dict, List

from src.integration.contracts import (
    CorrectedBlock,
    CorrectedDocument,
    CorrectedPage,
    DomainClassification,
    OCRDocument,
)
from src.integration.adapters.ocr_adapter import OCRAdapter, DomainAdapter
from src.correction.inference import CorrectionEngine


class IntegrationPipeline:
    """
    Coordinates the full end-to-end integration:
    OCR (Member 2) -> Domain Classification (Member 2) -> Correction (Member 3)
    """

    def __init__(self, correction_engine: CorrectionEngine):
        self.engine = correction_engine

    def process_document(
        self,
        raw_ocr_data: Dict[str, Any],
        raw_domain_data: Dict[str, Any],
    ) -> CorrectedDocument:
        """
        End-to-end processing pipeline for a single document.

        raw_ocr_data: dict matching OCRDocument schema (from Member 2 adapter
            or directly constructed for testing).
        raw_domain_data: dict matching DomainClassification schema (from
            Member 2 DomainAdapter or directly constructed).
        """
        # 1. Adapt OCR
        ocr_doc = OCRAdapter.parse_member2_ocr(raw_ocr_data)

        # 2. Adapt Domain Classification
        domain_class = DomainAdapter.parse_member2_domain(raw_domain_data)

        corrected_pages = []
        for page in ocr_doc.pages:
            corrected_blocks = []
            for block in page.blocks:
                # 3. Build correction request with full Member-2 metadata
                req = {
                    "text": block.text,
                    # Use language from the block (was hardcoded "unknown" before)
                    "language": block.language,
                    "domain": block.domain if block.domain else domain_class.domain,
                    # Optional metadata for the prompt (correction engine uses
                    # these if it has a template section for hints)
                    "domain_confidence": (
                        block.domain_confidence
                        if block.domain_confidence is not None
                        else domain_class.confidence
                    ),
                    "terminology_flags": block.terminology_flags,
                    "ocr_confidence": block.confidence,
                }

                result = self.engine.correct(req)

                corrected_block = CorrectedBlock(
                    block_id=block.block_id,
                    original_text=block.text,
                    corrected_text=result["corrected_text"],
                    changes=result["changes"],
                    bbox=block.bbox,
                    ocr_confidence=block.confidence,
                    language=result.get("metadata", {}).get("language", block.language),
                    domain=domain_class.domain,
                    correction_metadata=result.get("metadata", {}),
                )
                corrected_blocks.append(corrected_block)

            corrected_pages.append(
                CorrectedPage(page_number=page.page_number, blocks=corrected_blocks)
            )

        return CorrectedDocument(document_id=ocr_doc.document_id, pages=corrected_pages)
