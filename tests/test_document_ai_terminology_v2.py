"""
Tests for Terminology Database v2 (Task 11).
"""

from pathlib import Path
import pytest

from src.document_ai.domain.terminology import TerminologyDatabase


def test_terminology_database_term_counts():
    db = TerminologyDatabase.load()
    assert "deep_learning" in db.terms_by_domain
    assert "computer_science" in db.terms_by_domain
    assert "mathematics" in db.terms_by_domain

    assert len(db.terms_by_domain["deep_learning"]) >= 150
    assert len(db.terms_by_domain["computer_science"]) >= 150
    assert len(db.terms_by_domain["mathematics"]) >= 150


def test_terminology_find_protected_terms():
    db = TerminologyDatabase.load()

    text = "We use the transformer architecture with backpropagation and dropout to train the model."
    protected = db.find_protected_terms(text)

    assert "transformer" in protected
    assert "backpropagation" in protected
    assert "dropout" in protected


def test_terminology_fuzzy_matching_ocr_errors():
    db = TerminologyDatabase.load()

    text = "The transfrmer uses atention mechanisms for modeling."
    findings = db.check_text(text, exact_match_skip=False)

    error_terms = [f["closest_term"] for f in findings if f["status"] == "possible_ocr_error"]
    assert "attention" in error_terms or "transformer" in error_terms


def test_terminology_protected_aliases():
    db = TerminologyDatabase.load()

    text = "The self-attention mechanism is a key component."
    protected = db.find_protected_terms(text)
    assert "attention" in protected or "self-attention" in protected
