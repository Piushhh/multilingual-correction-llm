"""
Domain classification baseline.
Uses deterministic keyword and technical terminology matching to classify
text into 'deep_learning' or fallback to 'general'.

Note on confidence: The confidence returned is a deterministic heuristic score
derived from domain keyword density and match count. It is NOT a calibrated
ML posterior probability, as this is a transparent rule-based research baseline.
"""

from typing import Any, Dict, List, Optional
from src.document_ai.domain.terminology import DomainTerminologyDatabase


class DomainClassifier:
    """
    Classifies text into technical domains based on terminology matching.
    Returns structured results compatible with Member 3's DomainClassification contract.
    """

    def __init__(self, terminology_db: Optional[DomainTerminologyDatabase] = None):
        """
        Initialize the classifier.

        Args:
            terminology_db: Optional DomainTerminologyDatabase instance.
        """
        self.db = terminology_db or DomainTerminologyDatabase()

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify input text into a domain.

        Args:
            text: Input document or block text.

        Returns:
            Dict compatible with DomainClassification contract:
            {
                "domain": "deep_learning" | "general",
                "confidence": float,
                "alternatives": [{"domain": str, "confidence": float}]
            }
        """
        if not text or not text.strip():
            return {
                "domain": "general",
                "confidence": 1.0,
                "alternatives": [{"domain": "deep_learning", "confidence": 0.0}],
            }

        matched_terms = self.db.find_matching_terms(text, domain="deep_learning")

        if matched_terms:
            # Heuristic score: 0.65 base + 0.10 for each additional term (capped at 0.95)
            confidence = min(0.95, round(0.65 + 0.10 * (len(matched_terms) - 1), 2))
            alt_conf = round(1.0 - confidence, 2)
            return {
                "domain": "deep_learning",
                "confidence": confidence,
                "alternatives": [{"domain": "general", "confidence": alt_conf}],
            }
        else:
            # Fallback to general domain
            return {
                "domain": "general",
                "confidence": 0.85,
                "alternatives": [{"domain": "deep_learning", "confidence": 0.15}],
            }
