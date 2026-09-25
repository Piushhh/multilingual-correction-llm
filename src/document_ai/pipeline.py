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
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import ValidationError

from src.document_ai.ocr.extract import extract_regions, extract_full_text
from src.document_ai.ocr.preprocess import PreprocessConfig
from src.document_ai.language.detector import detect_language
from src.document_ai.domain.classifier import DomainClassifier
from src.document_ai.domain.terminology import TerminologyDatabase, DEFAULT_TERMINOLOGY_DIR
from src.document_ai.schemas import DocumentAIOutput, DocumentAIBatchOutput, INTERFACE_VERSION

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_PATH = Path("data/domain_classifier/classifier.joblib")


class PipelineOutputError(RuntimeError):
    """
    Raised when the DocumentAI pipeline produces output that violates
    the v1.0 interface schema. Chains the underlying pydantic ValidationError.
    """
    pass


def _try_load_default_classifier(classifier_path: Optional[Union[str, Path]] = None) -> Optional[DomainClassifier]:
    """Attempt to load the domain classifier from disk, failing gracefully if missing or incompatible."""
    target_path = Path(classifier_path) if classifier_path else DEFAULT_CLASSIFIER_PATH
    if not target_path.exists():
        return None
    try:
        return DomainClassifier.load(target_path)
    except Exception as exc:
        logger.warning(
            f"Failed to load domain classifier from {target_path} ({exc}); "
            "continuing without domain classification."
        )
        return None


def process_document(
    image_path,
    page_id,
    domain_classifier=None,
    terminology_db=None,
    languages="eng+hin",
    preprocess_config: Optional[PreprocessConfig] = None,
    no_domain: bool = False,
    min_domain_confidence: float = 0.40,
):
    """
    Run the full Member 2 pipeline on a single page image.

    domain_classifier: an optional DomainClassifier or path to one.
        If None and `no_domain=False`, attempts to load data/domain_classifier/classifier.joblib.
        If missing or `no_domain=True`, `domain`/`domain_confidence` are null.
    terminology_db: an optional TerminologyDatabase. If None, one is loaded
        from data/terminology/ (or is empty if that directory doesn't exist yet).
    preprocess_config: optional PreprocessConfig to override preprocessing defaults.
    min_domain_confidence: threshold below which domain is labelled as "general".

    Returns:
        A validated DocumentAIOutput dict (pydantic model .model_dump()).

    Raises:
        PipelineOutputError: if the pipeline produces output that violates
            the v1.0 schema (bug in pipeline code, not caller input).
    """
    regions, meta = extract_regions(
        image_path,
        languages=languages,
        config=preprocess_config,
        return_metadata=True,
    )
    full_text = extract_full_text(regions)

    language_result = detect_language(full_text)

    # Resolve domain classifier
    active_classifier = None
    if not no_domain:
        if isinstance(domain_classifier, DomainClassifier):
            active_classifier = domain_classifier
        elif isinstance(domain_classifier, (str, Path)):
            active_classifier = _try_load_default_classifier(domain_classifier)
        elif domain_classifier is None:
            active_classifier = _try_load_default_classifier()

    domain = None
    domain_confidence = None
    domain_scores = None
    if active_classifier is not None and full_text.strip():
        try:
            domain_result = active_classifier.predict(full_text)
            pred_domain = domain_result["domain"]
            pred_conf = domain_result["confidence"]
            domain_scores = domain_result.get("scores")

            if pred_conf < min_domain_confidence:
                domain = "general"
                domain_confidence = pred_conf
            else:
                domain = pred_domain
                domain_confidence = pred_conf
        except Exception as exc:
            logger.warning(f"Domain classifier prediction failed: {exc}")
            domain = None
            domain_confidence = None

    if terminology_db is None:
        terminology_db = TerminologyDatabase.load(DEFAULT_TERMINOLOGY_DIR)

    output_regions = []
    for region in regions:
        flags = terminology_db.check_text(region["text"])
        # Per-region language detection with length guard (>= 8 chars to avoid noisy fragments)
        reg_text = region["text"].strip()
        reg_lang = None
        if len(reg_text) >= 8:
            reg_lang = detect_language(reg_text)["language"]

        # Collect protected terms found in this region
        protected_in_reg = []
        if hasattr(terminology_db, "find_protected_terms"):
            protected_in_reg = terminology_db.find_protected_terms(reg_text)

        output_regions.append(
            {
                "text": region["text"],
                "bbox": region["bbox"],
                "confidence": region["confidence"],
                "terminology_flags": flags,
                "words": region.get("words"),
                "language": reg_lang,
                "protected_terms": protected_in_reg,
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
        "page_width": meta.get("original_width"),
        "page_height": meta.get("original_height"),
        "domain_scores": domain_scores,
    }

    # Validate against the schema — catches pipeline bugs
    try:
        validated = DocumentAIOutput(**raw_output)
    except ValidationError as exc:
        raise PipelineOutputError(
            f"[DocumentAI pipeline] Output failed schema validation "
            f"(this is a pipeline bug, not a caller error):\n{exc}"
        ) from exc

    return validated.model_dump()


def process_pdf(
    pdf_path,
    document_id=None,
    dpi=200,
    domain_classifier=None,
    terminology_db=None,
    languages="eng+hin",
    preprocess_config: Optional[PreprocessConfig] = None,
    no_domain: bool = False,
    min_domain_confidence: float = 0.40,
) -> Dict[str, Any]:
    """
    Process a PDF file page-by-page using PyMuPDF (fitz).
    Renders each page to an image and runs process_document, returning
    a validated DocumentAIBatchOutput dict with page IDs doc_p001, doc_p002, etc.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for process_pdf. Install it with: pip install pymupdf"
        ) from exc

    import os
    import tempfile

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = document_id or pdf_file.stem
    doc = fitz.open(str(pdf_file))
    pages_output = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_id = f"{doc_id}_p{page_num:03d}"

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            pix.save(tmp_path)
            page_res = process_document(
                image_path=tmp_path,
                page_id=page_id,
                domain_classifier=domain_classifier,
                terminology_db=terminology_db,
                languages=languages,
                preprocess_config=preprocess_config,
                no_domain=no_domain,
                min_domain_confidence=min_domain_confidence,
            )
            pages_output.append(page_res)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    batch_output = {
        "version": INTERFACE_VERSION,
        "document_id": doc_id,
        "page_count": len(pages_output),
        "pages": pages_output,
    }

    try:
        validated = DocumentAIBatchOutput(**batch_output)
    except ValidationError as exc:
        raise PipelineOutputError(
            f"[DocumentAI pipeline] Batch output failed schema validation:\n{exc}"
        ) from exc

    return validated.model_dump()


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Member 2 DocumentAI CLI — process a document image or PDF and write "
            "structured JSON output."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", metavar="PATH",
        help="Path to the input image (JPEG, PNG, TIFF, etc.).",
    )
    group.add_argument(
        "--pdf", metavar="PATH",
        help="Path to input multi-page PDF document.",
    )
    parser.add_argument(
        "--out", default="-", metavar="PATH",
        help="Output path for the JSON result. Defaults to stdout ('-').",
    )
    parser.add_argument(
        "--page-id", default=None, metavar="ID",
        help="Document / page identifier to embed in the output. Defaults to filename stem.",
    )
    parser.add_argument(
        "--lang", default="eng+hin", metavar="LANGS",
        help="Tesseract language codes (default: eng+hin).",
    )
    parser.add_argument(
        "--classifier", default=None, metavar="PATH",
        help="Path to trained domain classifier joblib model (default: data/domain_classifier/classifier.joblib).",
    )
    parser.add_argument(
        "--no-domain", action="store_true",
        help="Disable domain classification.",
    )
    parser.add_argument(
        "--no-terminology", action="store_true",
        help="Skip terminology checking (faster, for debugging).",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    terminology_db = TerminologyDatabase({}) if args.no_terminology else None

    try:
        if args.pdf:
            result = process_pdf(
                pdf_path=args.pdf,
                document_id=args.page_id,
                domain_classifier=args.classifier,
                no_domain=args.no_domain,
                terminology_db=terminology_db,
                languages=args.lang,
            )
        else:
            image_path = Path(args.image)
            if not image_path.exists():
                print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
                sys.exit(1)
            page_id = args.page_id or image_path.stem
            result = process_document(
                image_path=str(image_path),
                page_id=page_id,
                domain_classifier=args.classifier,
                no_domain=args.no_domain,
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
