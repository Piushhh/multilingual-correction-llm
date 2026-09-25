"""
DocumentAI Interface Schema — version 1.0

Pydantic models for the structured output of process_document().
These models are the formal contract between Member 2 (document_ai)
and all downstream consumers (Member 3 correction engine, integration layer).

Interface versioning policy:
  - INTERFACE_VERSION = "1.0"
  - Additive changes (new optional fields): keep "1.0"
  - Breaking changes (removed/renamed fields, type changes): bump to "2.0"
    and update CHANGELOG below, all tests, and all adapters.

CHANGELOG:
  1.0.1 (additive):
    - DocumentRegion: words, language, protected_terms
    - DocumentAIOutput: page_width, page_height, domain_scores
    - DocumentAIBatchOutput: multi-page batch container
  1.0 (initial): page_id, language, language_confidence, domain,
                 domain_confidence, regions[text, bbox, confidence,
                 terminology_flags]
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


INTERFACE_VERSION = "1.0"

# Allowed language codes (Member 2 contract)
_VALID_LANGUAGES = {"en", "hi", "code_mixed", "unknown"}

# Allowed domain labels (None means classifier not run)
_VALID_DOMAINS = {"deep_learning", "computer_science", "mathematics", "general"}


class TerminologyFlag(BaseModel):
    """One flag entry from the terminology checker."""
    word: str
    position: int = Field(ge=0)
    status: str  # "verified_term" | "possible_ocr_error"
    closest_term: str
    similarity: float = Field(ge=0.0, le=100.0)


class DocumentRegion(BaseModel):
    """A single text region extracted from a page image."""
    text: str
    bbox: List[int] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    terminology_flags: List[TerminologyFlag] = []
    # Additive optional fields (v1.0.1)
    words: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    protected_terms: List[str] = []

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_valid(cls, v: List[int]) -> List[int]:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 elements: [x_min, y_min, x_max, y_max]")
        x_min, y_min, x_max, y_max = v
        if x_max < x_min or y_max < y_min:
            raise ValueError(
                f"bbox invalid: x_max ({x_max}) must be >= x_min ({x_min}) "
                f"and y_max ({y_max}) must be >= y_min ({y_min})"
            )
        return v


class DocumentAIOutput(BaseModel):
    """
    The full structured output of process_document().

    version:             always INTERFACE_VERSION — allows consumers to detect
                         schema changes programmatically.
    page_id:             caller-supplied identifier for the page/image.
    language:            one of 'en', 'hi', 'code_mixed', 'unknown'.
    language_confidence: 0.0–1.0; 0.0 means no text was found.
    domain:              None when no classifier was passed; otherwise one of
                         'deep_learning', 'computer_science', 'mathematics',
                         'general'.
    domain_confidence:   None when domain is None; otherwise 0.0–1.0.
    regions:             list of text regions in reading order.
    page_width:          optional original page width in pixels.
    page_height:         optional original page height in pixels.
    domain_scores:       optional dict mapping domain labels to probabilities.
    """

    version: str = INTERFACE_VERSION
    page_id: str
    language: str
    language_confidence: float = Field(ge=0.0, le=1.0)
    domain: Optional[str] = None
    domain_confidence: Optional[float] = None
    regions: List[DocumentRegion]
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    domain_scores: Optional[Dict[str, float]] = None

    @field_validator("language")
    @classmethod
    def language_must_be_valid(cls, v: str) -> str:
        if v not in _VALID_LANGUAGES:
            raise ValueError(
                f"language '{v}' not in allowed set {sorted(_VALID_LANGUAGES)}"
            )
        return v

    @field_validator("domain")
    @classmethod
    def domain_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_DOMAINS:
            raise ValueError(
                f"domain '{v}' not in allowed set {sorted(_VALID_DOMAINS)}"
            )
        return v

    @model_validator(mode="after")
    def domain_confidence_consistency(self) -> "DocumentAIOutput":
        if self.domain is None and self.domain_confidence is not None:
            raise ValueError("domain_confidence must be None when domain is None")
        if self.domain is not None and self.domain_confidence is None:
            raise ValueError("domain_confidence must be set when domain is not None")
        if self.domain_confidence is not None and not (0.0 <= self.domain_confidence <= 1.0):
            raise ValueError("domain_confidence must be in [0.0, 1.0]")
        return self


class DocumentAIBatchOutput(BaseModel):
    """
    Multi-page document output container (e.g. from PDF processing).
    """
    version: str = INTERFACE_VERSION
    document_id: str
    page_count: int = Field(ge=0)
    pages: List[DocumentAIOutput]


def export_json_schema(output_path: str = None) -> Dict[str, Any]:
    """
    Export the DocumentAIOutput schema as a JSON Schema dict.

    If output_path is given, writes the schema to that file.
    Always returns the schema dict.
    """
    schema = DocumentAIOutput.model_json_schema()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
    return schema


if __name__ == "__main__":
    # Generate and print the JSON schema
    schema = export_json_schema("data/schemas/document_ai_schema.json")
    print(json.dumps(schema, indent=2))
