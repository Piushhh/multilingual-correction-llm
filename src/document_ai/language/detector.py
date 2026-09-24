"""
Script-based deterministic language detector.
Supports English ('en'), Hindi ('hi'), and Code-Mixed ('code-mixed') text
using Unicode character block analysis without external ML dependencies.
"""

import unicodedata
from typing import Any, Dict


class LanguageDetector:
    """
    Deterministic language detector using script and Unicode character distributions.

    Note: This is a script-based baseline and distinguishes English (Latin script),
    Hindi (Devanagari script), and code-mixed text. It does not perform full
    morphosyntactic linguistic modeling.
    """

    @staticmethod
    def _is_latin(char: str) -> bool:
        """Check if character is Latin alphabet."""
        return "LATIN" in unicodedata.name(char, "")

    @staticmethod
    def _is_devanagari(char: str) -> bool:
        """Check if character is Devanagari script."""
        return "DEVANAGARI" in unicodedata.name(char, "")

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect language of input text.

        Args:
            text: Input string.

        Returns:
            Dict containing 'language' and 'confidence':
            {"language": "en" | "hi" | "code-mixed" | "unknown", "confidence": float}
        """
        if not text or not text.strip():
            return {"language": "unknown", "confidence": 0.0}

        latin_count = 0
        devanagari_count = 0
        other_alpha_count = 0

        for ch in text:
            if not ch.isalpha():
                continue
            if self._is_latin(ch):
                latin_count += 1
            elif self._is_devanagari(ch):
                devanagari_count += 1
            else:
                other_alpha_count += 1

        total_alpha = latin_count + devanagari_count + other_alpha_count

        if total_alpha == 0:
            return {"language": "unknown", "confidence": 0.0}

        latin_ratio = latin_count / total_alpha
        devanagari_ratio = devanagari_count / total_alpha

        # Code-mixed detection: both scripts present with meaningful contribution
        if latin_count > 0 and devanagari_count > 0:
            # If both scripts are present with at least 10% representation or >=2 characters
            if min(latin_ratio, devanagari_ratio) >= 0.10 or min(latin_count, devanagari_count) >= 2:
                # Confidence is higher when both scripts are substantially represented
                mixed_balance = 1.0 - abs(latin_ratio - devanagari_ratio)
                confidence = round(0.70 + 0.30 * mixed_balance, 2)
                return {"language": "code-mixed", "confidence": confidence}

        # Dominant English (Latin script)
        if latin_ratio > 0.85:
            return {"language": "en", "confidence": round(latin_ratio, 2)}

        # Dominant Hindi (Devanagari script)
        if devanagari_ratio > 0.85:
            return {"language": "hi", "confidence": round(devanagari_ratio, 2)}

        # Plurality fallback
        if latin_count > devanagari_count:
            return {"language": "en", "confidence": round(latin_ratio, 2)}
        elif devanagari_count > latin_count:
            return {"language": "hi", "confidence": round(devanagari_ratio, 2)}
        else:
            return {"language": "code-mixed", "confidence": 0.75}

    def detect_language(self, text: str) -> str:
        """Convenience helper returning just the detected language string."""
        return self.detect(text)["language"]


# Convenience singleton function
_default_detector = LanguageDetector()


def detect_language(text: str) -> Dict[str, Any]:
    """Helper function to detect language using default detector."""
    return _default_detector.detect(text)
