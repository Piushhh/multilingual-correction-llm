"""
Tests for PDF processing and multi-page adapter integration (Task 8 / Problem K).
"""

from pathlib import Path
import fitz  # PyMuPDF
import pytest

from src.document_ai.pipeline import process_pdf
from src.document_ai.schemas import DocumentAIBatchOutput
from src.integration.adapters.document_ai_adapter import adapt_document_ai_to_ocr_document


def _create_test_pdf(pdf_path: Path, page_count: int = 2):
    """Generate a clean synthetic multi-page PDF document."""
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page(width=595, height=842)  # A4 size in points
        # Insert some text on each page
        rect = fitz.Rect(50, 50, 500, 200)
        page.insert_textbox(rect, f"Page {i + 1} content: Deep learning and artificial intelligence study.", fontsize=18)
    doc.save(str(pdf_path))
    doc.close()


def test_process_pdf_generates_batch_output(tmp_path):
    pdf_path = tmp_path / "sample_doc.pdf"
    _create_test_pdf(pdf_path, page_count=2)

    result = process_pdf(
        pdf_path=str(pdf_path),
        document_id="sample_doc",
        no_domain=True,
    )

    assert result["document_id"] == "sample_doc"
    assert result["page_count"] == 2
    assert len(result["pages"]) == 2

    assert result["pages"][0]["page_id"] == "sample_doc_p001"
    assert result["pages"][1]["page_id"] == "sample_doc_p002"
    assert result["pages"][0]["version"] == "1.0"
    assert result["pages"][1]["version"] == "1.0"


def test_adapt_batch_output_to_multi_page_ocr_document(tmp_path):
    pdf_path = tmp_path / "two_pages.pdf"
    _create_test_pdf(pdf_path, page_count=2)

    batch_output = process_pdf(
        pdf_path=str(pdf_path),
        document_id="two_pages",
        no_domain=True,
    )

    ocr_doc = adapt_document_ai_to_ocr_document(batch_output)

    assert ocr_doc.document_id == "two_pages"
    assert len(ocr_doc.pages) == 2
    assert ocr_doc.pages[0].page_number == 1
    assert ocr_doc.pages[1].page_number == 2


def test_adapt_single_page_output_backward_compatible():
    single_page_data = {
        "version": "1.0",
        "page_id": "test_single_page",
        "language": "en",
        "language_confidence": 0.95,
        "domain": "deep_learning",
        "domain_confidence": 0.85,
        "regions": [
            {
                "text": "Deep neural network optimization",
                "bbox": [50, 50, 300, 80],
                "confidence": 0.92,
                "terminology_flags": [],
            }
        ],
    }

    ocr_doc = adapt_document_ai_to_ocr_document(single_page_data)

    assert ocr_doc.document_id == "test_single_page"
    assert len(ocr_doc.pages) == 1
    assert ocr_doc.pages[0].page_number == 1
    assert len(ocr_doc.pages[0].blocks) == 1
    assert ocr_doc.pages[0].blocks[0].block_id == "p1_r0"
    assert ocr_doc.pages[0].blocks[0].domain == "deep_learning"
