from typing import Dict, Any
from src.integration.contracts import OCRDocument, DomainClassification

class OCRAdapter:
    """Adapts raw OCR outputs to the Correction pipeline standard."""
    
    @staticmethod
    def parse_member1_ocr(raw_data: Dict[str, Any]) -> OCRDocument:
        """
        Parses Member 1's OCR structure into the pipeline OCRDocument contract.
        Since Member 1's exact structure is unknown, this is a clean interface mapping.
        """
        # Mock logic adapting an unknown format to our contract
        return OCRDocument(**raw_data)

class DomainAdapter:
    """Adapts domain classification to the Correction pipeline standard."""
    
    @staticmethod
    def parse_member2_domain(raw_data: Dict[str, Any]) -> DomainClassification:
        """
        Parses Member 2's Domain Classifier into the standard contract.
        """
        if not raw_data or "domain" not in raw_data:
            return DomainClassification(domain="general", confidence=1.0)
            
        # fallback to general domain if confidence is low
        if raw_data.get("confidence", 1.0) < 0.5:
            return DomainClassification(domain="general", confidence=raw_data.get("confidence", 1.0))
            
        return DomainClassification(**raw_data)
