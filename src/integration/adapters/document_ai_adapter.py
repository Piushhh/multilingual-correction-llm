"""
DocumentAI -> OCRDocument Adapter (Stage 4)

Converts the Member 2 DocumentAI v1.0 output dict into the Member 3
OCRDocument contract, without rewriting Member 2 code.

Mapping:
    DocumentAI.page_id              -> document_id (base), page_number=1
    DocumentAI.regions[i]           -> OCRBlock with id "p1_r{i}"
    DocumentAI.regions[i].text      -> OCRBlock.text
    DocumentAI.regions[i].bbox      -> OCRBlock.bbox
    DocumentAI.regions[i].confidence-> OCRBlock.confidence
    DocumentAI.language             -> OCRBlock.language (mapped to human name)
    DocumentAI.language_confidence  -> OCRBlock.language_confidence
    DocumentAI.domain               -> OCRBlock.domain (None -> "general")
    DocumentAI.domain_confidence    -> OCRBlock.domain_confidence
    DocumentAI.regions[i].terminology_flags -> OCRBlock.terminology_flags

Does NOT drop information: all fields from DocumentAI output are carried
through or mapped. Validates input against DocumentAI schema before converting.

Language code mapping (Member 2 -> human name for Member 3 prompt):
    "en"         -> "English"
    "hi"         -> "Hindi"
    "code_mixed" -> "English-Hindi code-mixed"
    "unknown"    -> "unknown"
"""

from typing import Any, Dict

from pydantic import ValidationError

from src.document_ai.schemas import DocumentAIOutput
from src.integration.contracts import OCRBlock, OCRDocument, OCRPage


_LANGUAGE_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "code_mixed": "English-Hindi code-mixed",
    "unknown": "unknown",
}


def configure_easyocr_reader():
    """
    Configure and return an EasyOCR Reader instance with the languages
    and settings defined by Khushi's OCR integration.
    """
    try:
        import easyocr
        return easyocr.Reader(
            ["en", "hi"],
            gpu=False,
        )
    except ImportError:
        import logging
        logging.getLogger(__name__).warning("easyocr is not installed")
        return None


def adapt_document_ai_to_ocr_document(document_ai_output: Dict[str, Any]) -> OCRDocument:
    """
    Convert a Member 2 DocumentAI v1.0 output dict into a Member 3 OCRDocument.

    Args:
        document_ai_output: The dict returned by process_document() — already
            validated by Member 2's pipeline. This adapter validates it again
            as a defensive measure.

    Returns:
        OCRDocument ready for IntegrationPipeline.process_document().

    Raises:
        ValueError: if the input fails DocumentAI schema validation.
    """
    # Defensive re-validation against the v1.0 schema
    try:
        validated = DocumentAIOutput(**document_ai_output)
    except (ValidationError, Exception) as exc:
        raise ValueError(
            f"document_ai_to_ocr_document: input failed DocumentAI v1.0 "
            f"schema validation. Ensure the dict came from process_document().\n{exc}"
        ) from exc

    page_id = validated.page_id
    language_code = validated.language
    language_name = _LANGUAGE_CODE_TO_NAME.get(language_code, "unknown")
    language_confidence = validated.language_confidence

    # Domain: None -> "general" (matching DomainAdapter behaviour)
    domain = validated.domain if validated.domain is not None else "general"
    domain_confidence = validated.domain_confidence

    blocks = []
    for i, region in enumerate(validated.regions):
        block_id = f"p1_r{i}"
        blocks.append(
            OCRBlock(
                block_id=block_id,
                text=region.text,
                bbox=region.bbox,
                confidence=region.confidence,
                language=language_name,
                language_confidence=language_confidence,
                domain=domain,
                domain_confidence=domain_confidence,
                terminology_flags=[f.model_dump() for f in region.terminology_flags],
            )
        )

    page = OCRPage(page_number=1, blocks=blocks)
    return OCRDocument(document_id=page_id, pages=[page])
