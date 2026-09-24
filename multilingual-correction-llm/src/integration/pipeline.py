from typing import Dict, Any, List
from src.integration.contracts import OCRDocument, DomainClassification, CorrectedDocument, CorrectedPage, CorrectedBlock
from src.integration.adapters.ocr_adapter import OCRAdapter, DomainAdapter
from src.correction.inference import CorrectionEngine

class IntegrationPipeline:
    """
    Coordinates the full end-to-end integration:
    OCR (Member 1) -> Domain Classification (Member 2) -> Correction (Member 3)
    """
    def __init__(self, correction_engine: CorrectionEngine):
        self.engine = correction_engine
        
    def process_document(self, raw_ocr_data: Dict[str, Any], raw_domain_data: Dict[str, Any]) -> CorrectedDocument:
        """
        End-to-end processing pipeline for a single document.
        """
        # 1. Adapt OCR
        ocr_doc = OCRAdapter.parse_member1_ocr(raw_ocr_data)
        
        # 2. Adapt Domain Classification
        domain_class = DomainAdapter.parse_member2_domain(raw_domain_data)
        
        corrected_pages = []
        for page in ocr_doc.pages:
            corrected_blocks = []
            
            for block in page.blocks:
                # 3. Correction Inference
                req = {
                    "text": block.text,
                    "language": "unknown", # To be determined by language identification if needed
                    "domain": domain_class.domain
                }
                
                # We enforce inference safety 
                result = self.engine.correct(req)
                
                corrected_block = CorrectedBlock(
                    block_id=block.block_id,
                    original_text=block.text,
                    corrected_text=result["corrected_text"],
                    changes=result["changes"],
                    bbox=block.bbox,
                    ocr_confidence=block.confidence,
                    language=result.get("metadata", {}).get("language", "unknown"),
                    domain=domain_class.domain,
                    correction_metadata=result.get("metadata", {}),
                )
                corrected_blocks.append(corrected_block)
                
            corrected_pages.append(CorrectedPage(page_number=page.page_number, blocks=corrected_blocks))
            
        return CorrectedDocument(document_id=ocr_doc.document_id, pages=corrected_pages)
