"""
End-to-end Member 2 pipeline: image -> structured document understanding
output.

This is the "Structured document understanding output" milestone (M7) and
implements the exact interface the team agreed on (roadmap section 5,
Interface 2), extended with a `version` field and per-region diagnostics so
it can evolve without silently breaking Member 1/3's integration code.

Interface (version "1.0"):
{
    "version": "1.0",
    "page_id": str,
    "language": "en" | "hi" | "code_mixed" | "unknown",
    "language_confidence": float,
    "domain": "deep_learning" | "computer_science" | "mathematics" | "general" | null,
    "domain_confidence": float | null,
    "regions": [
        {
            "text": str,
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": float,          # OCR confidence, 0-1
            "terminology_flags": [ ... ]  # from terminology.check_text, may be []
        },
        ...
    ]
}

`domain` is null when no trained classifier is supplied -- OCR + language
detection work standalone (useful for Member 3 to develop against before
the classifier is trained), domain classification is an opt-in extra step.
"""

from src.document_ai.ocr.extract import extract_regions, extract_full_text
from src.document_ai.language.detector import detect_language
from src.document_ai.domain.terminology import TerminologyDatabase, DEFAULT_TERMINOLOGY_DIR

INTERFACE_VERSION = "1.0"


def process_document(
    image_path,
    page_id,
    domain_classifier=None,
    terminology_db=None,
    languages="eng+hin",
):
    """
    Run the full Member 2 pipeline on a single page image.

    domain_classifier: an optional src.document_ai.domain.classifier.DomainClassifier
        (already loaded/trained). If None, `domain`/`domain_confidence` are null.
    terminology_db: an optional TerminologyDatabase. If None, one is loaded
        from data/terminology/ (or is empty if that directory doesn't exist yet).
    """
    regions = extract_regions(image_path, languages=languages)
    full_text = extract_full_text(regions)

    language_result = detect_language(full_text)

    domain = None
    domain_confidence = None
    if domain_classifier is not None and full_text.strip():
        domain_result = domain_classifier.predict(full_text)
        domain = domain_result["domain"]
        domain_confidence = domain_result["confidence"]

    if terminology_db is None:
        terminology_db = TerminologyDatabase.load(DEFAULT_TERMINOLOGY_DIR)

    output_regions = []
    for region in regions:
        flags = terminology_db.check_text(region["text"])
        output_regions.append(
            {
                "text": region["text"],
                "bbox": region["bbox"],
                "confidence": region["confidence"],
                "terminology_flags": flags,
            }
        )

    return {
        "version": INTERFACE_VERSION,
        "page_id": page_id,
        "language": language_result["language"],
        "language_confidence": language_result["confidence"],
        "domain": domain,
        "domain_confidence": domain_confidence,
        "regions": output_regions,
    }
