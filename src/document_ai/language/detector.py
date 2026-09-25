"""
Language detection for English, Hindi, and code-mixed text.

Member 2, Responsibility 3: "Detect English, Hindi, and code-mixed text
where feasible."

Two signals are combined rather than relying on `langdetect` alone:

1. Script composition (Devanagari-range vs. Latin-range character counts).
   This is cheap, has zero failure modes on short/noisy OCR output (unlike
   statistical detectors, which need a reasonable amount of text), and is
   exactly what distinguishes Hindi and English at the character level.
2. `langdetect` (a Python port of Google's language-detection library) as a
   secondary check on the Latin-script portion, since script alone can't
   tell English from other Latin-script languages.

"Code-mixed" is reported when both scripts are present above a small
threshold -- this is the common case for OCR'd technical documents in
India, where English technical terms appear inline in Hindi sentences (or
vice versa), and it's explicitly called out as a first-class case in the
project scope.
"""

import re

try:
    from langdetect import DetectorFactory, LangDetectException
    from langdetect import detect as _langdetect_detect

    # Force determinism in statistical language detection (Task 5)
    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only if dependency missing
    _LANGDETECT_AVAILABLE = False


# Unicode block for Devanagari (covers Hindi, Marathi, Sanskrit, etc.)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# Basic Latin letters, used as the "English/Latin-script" character count.
_LATIN_RE = re.compile(r"[A-Za-z]")


def script_composition(text):
    """
    Return {"devanagari": int, "latin": int, "other": int} character counts.
    Whitespace, punctuation, and digits are not counted in any bucket --
    they're script-neutral and would dilute the ratio.
    """
    devanagari = len(_DEVANAGARI_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    letters_total = len(re.findall(r"\w", text, flags=re.UNICODE))
    other = max(0, letters_total - devanagari - latin)

    return {"devanagari": devanagari, "latin": latin, "other": other}


def detect_language(text, code_mixed_threshold=0.15):
    """
    Classify `text` as one of: "en", "hi", "code_mixed", or "unknown".

    code_mixed_threshold: the minority script must make up at least this
    fraction of (devanagari + latin) characters for the text to be called
    code-mixed rather than purely one language. 0.15 means a document
    that's ~85% Hindi with a sprinkling of English terms is still called
    code_mixed, which matches how these technical documents actually read.

    Returns a dict: {"language": str, "confidence": float, "scripts": {...}}
    """
    composition = script_composition(text)
    devanagari = composition["devanagari"]
    latin = composition["latin"]
    script_total = devanagari + latin

    if script_total == 0:
        return {"language": "unknown", "confidence": 0.0, "scripts": composition}

    devanagari_ratio = devanagari / script_total
    latin_ratio = latin / script_total

    minority_ratio = min(devanagari_ratio, latin_ratio)

    if devanagari > 0 and latin > 0 and minority_ratio >= code_mixed_threshold:
        # Confidence here reflects how balanced the mix is -- a near-50/50
        # split is a more confident "code_mixed" call than a 95/5 split
        # that only just crossed the threshold.
        confidence = round(1.0 - abs(devanagari_ratio - latin_ratio), 4)
        return {
            "language": "code_mixed",
            "confidence": confidence,
            "scripts": composition,
        }

    if devanagari_ratio >= latin_ratio:
        return {
            "language": "hi",
            "confidence": round(devanagari_ratio, 4),
            "scripts": composition,
        }

    # Latin-dominant: cross-check with langdetect when available and there's
    # enough text for it to be reliable (it's unstable on very short
    # strings, which OCR line-regions often are).
    confidence = round(latin_ratio, 4)

    if _LANGDETECT_AVAILABLE and len(text.strip()) >= 20:
        try:
            statistical_guess = _langdetect_detect(text)
            if statistical_guess != "en":
                # Latin-script but not English (e.g. accidental match on
                # another European language) -- flag it rather than
                # silently mislabel, since "en" is the only Latin-script
                # language this project targets.
                return {
                    "language": "unknown",
                    "confidence": confidence,
                    "scripts": composition,
                    "note": f"langdetect guessed '{statistical_guess}', not 'en'",
                }
        except LangDetectException:
            pass  # too little/ambiguous text for langdetect; trust the script signal

    return {"language": "en", "confidence": confidence, "scripts": composition}


def detect_document_language(regions):
    """
    Detect the overall document language from a list of OCR regions (as
    produced by ocr.extract.extract_regions), by running detection on the
    concatenated page text.

    Region-level language is intentionally NOT computed separately here --
    per-line detection is noisy on short OCR lines. Callers that need
    per-region granularity should call `detect_language` directly on
    individual region text and treat short-line results with lower trust.
    """
    full_text = " ".join(r["text"] for r in regions)
    return detect_language(full_text)
