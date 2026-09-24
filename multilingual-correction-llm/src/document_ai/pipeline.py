"""
End-to-end Member 2 pipeline: image -> structured document understanding output.

Interface version: "1.0" (see src/document_ai/schemas.py for full spec)

Structured output shape:
{
    "version": "1.0",
    "page_id": str,
    "language": "en" | "hi" | "code_mixed" | "unknown",
    "language_confidence": float,
    "domain": str | null,       # null when no classifier passed
    "domain_confidence": float | null,
    "regions": [
        {
            "text": str,
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": float,
            "terminology_flags": [...]
        }
    ]
}

domain is null when no trained classifier is supplied — OCR + language
detection work standalone (useful for Member 3 to develop against before
the classifier is trained).
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.document_ai.ocr.extract import extract_regions, extract_full_text
from src.document_ai.language.detector import detect_language
from src.document_ai.domain.terminology import TerminologyDatabase, DEFAULT_TERMINOLOGY_DIR
from src.document_ai.schemas import DocumentAIOutput, INTERFACE_VERSION


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

    Returns:
        A validated DocumentAIOutput dict (pydantic model .model_dump()).

    Raises:
        pydantic.ValidationError: if the pipeline produces output that violates
            the v1.0 schema (bug in pipeline code, not in caller input).
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

    raw_output = {
        "version": INTERFACE_VERSION,
        "page_id": page_id,
        "language": language_result["language"],
        "language_confidence": language_result["confidence"],
        "domain": domain,
        "domain_confidence": domain_confidence,
        "regions": output_regions,
    }

    # Validate against the schema — raises ValidationError if the pipeline
    # itself produces bad data (this is a bug-catching gate, not input validation).
    try:
        validated = DocumentAIOutput(**raw_output)
    except ValidationError as exc:
        raise ValidationError(
            f"[DocumentAI pipeline] Output failed schema validation "
            f"(this is a pipeline bug, not a caller error):\n{exc}"
        ) from exc

    return validated.model_dump()


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Member 2 DocumentAI CLI — process a document image and write "
            "structured JSON output."
        )
    )
    parser.add_argument(
        "--image", required=True, metavar="PATH",
        help="Path to the input image (JPEG, PNG, TIFF, etc.).",
    )
    parser.add_argument(
        "--out", default="-", metavar="PATH",
        help=(
            "Output path for the JSON result. "
            "Defaults to stdout ('-')."
        ),
    )
    parser.add_argument(
        "--page-id", default=None, metavar="ID",
        help=(
            "Page identifier to embed in the output. "
            "Defaults to the image filename stem."
        ),
    )
    parser.add_argument(
        "--lang", default="eng+hin", metavar="LANGS",
        help="Tesseract language codes (default: eng+hin).",
    )
    parser.add_argument(
        "--no-terminology", action="store_true",
        help="Skip terminology checking (faster, for debugging).",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    page_id = args.page_id or image_path.stem

    terminology_db = TerminologyDatabase({}) if args.no_terminology else None

    try:
        result = process_document(
            image_path=str(image_path),
            page_id=page_id,
            terminology_db=terminology_db,
            languages=args.lang,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    if args.out == "-":
        print(json_str)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"Output written to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
