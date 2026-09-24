"""
Domain Terminology Database.
Provides seed glossaries for technical domains (Deep Learning), term lookup,
normalization, and glossary membership validation.
"""

import re
from typing import Dict, List, Optional, Sequence, Set


# Documented seed terminology baseline for Deep Learning domain.
# These terms reflect the project corpus (English and Hindi equivalents).
DEEP_LEARNING_SEED_TERMS: Set[str] = {
    # English terms
    "transformer",
    "attention",
    "self-attention",
    "multi-head attention",
    "neural network",
    "deep learning",
    "machine learning",
    "convolutional",
    "backpropagation",
    "gradient descent",
    "embedding",
    "layer",
    "loss function",
    "learning rate",
    "epoch",
    "weights",
    "bias",
    "overfitting",
    "activation",
    "logits",
    "token",
    "tokenizer",
    "bpe",
    "perplexity",
    "cross-entropy",
    "feedforward",
    "residual",
    # Hindi equivalents/transliterations from project corpus
    "डीप लर्निंग",
    "मशीन लर्निंग",
    "न्यूरल नेटवर्क",
    "अटेंशन",
    "बैकप्रोपेगेशन",
    "ग्रेडिएंट",
    "ग्रेडिएंट डिसेंट",
    "लॉस फ़ंक्शन",
    "लॉस फक्शन",
    "लर्निंग रेट",
    "कन्वोल्यूशनल",
    "ट्रांसफॉर्मर",
}


class DomainTerminologyDatabase:
    """
    Manages domain-specific technical terminology glossaries.
    Supports term lookup, normalization, and preservation checks.
    """

    def __init__(self, custom_glossaries: Optional[Dict[str, Set[str]]] = None):
        """
        Initialize the terminology database.

        Args:
            custom_glossaries: Optional dict mapping domain_name -> set of term strings.
        """
        self._glossaries: Dict[str, Set[str]] = {
            "deep_learning": set(t.lower() for t in DEEP_LEARNING_SEED_TERMS),
            "general": set(),
        }
        if custom_glossaries:
            for domain, terms in custom_glossaries.items():
                self._glossaries[domain] = set(t.lower() for t in terms)

    def normalize_term(self, term: str) -> str:
        """Normalize term by stripping punctuation and lowercasing."""
        if not term:
            return ""
        cleaned = re.sub(r"[^\w\s\u0900-\u097F\-]", "", term.strip().lower())
        return re.sub(r"\s+", " ", cleaned)

    def is_domain_term(self, term: str, domain: str = "deep_learning") -> bool:
        """Check whether a term belongs to the specified domain glossary."""
        norm = self.normalize_term(term)
        if not norm or domain not in self._glossaries:
            return False
        return norm in self._glossaries[domain]

    def find_matching_terms(self, text: str, domain: str = "deep_learning") -> List[str]:
        """
        Find all domain terms present in the text.
        """
        norm_text = self.normalize_term(text)
        if not norm_text or domain not in self._glossaries:
            return []

        matched = []
        glossary = self._glossaries[domain]
        # Match multi-word terms first, then single-word terms
        sorted_terms = sorted(glossary, key=lambda t: len(t.split()), reverse=True)

        for term in sorted_terms:
            # Use word boundary matching for Latin words, substring for Devanagari
            pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
            if re.search(pattern, norm_text):
                matched.append(term)

        return matched

    def get_terms(self, domain: str = "deep_learning") -> Set[str]:
        """Return the complete set of terms for a domain."""
        return set(self._glossaries.get(domain, set()))

    def add_terms(self, domain: str, terms: Sequence[str]) -> None:
        """Add new terms to a domain glossary."""
        if domain not in self._glossaries:
            self._glossaries[domain] = set()
        for t in terms:
            self._glossaries[domain].add(self.normalize_term(t))
