"""
src.evaluation — Evaluation metrics package (Member 3).

Public exports:
    - compute_character_error_rate: CER metric
    - compute_word_error_rate: WER metric
    - compute_exact_match: Exact match metric
    - evaluate_metrics: All basic correction metrics
    - evaluate_edits: Edit-level precision/recall
    - evaluate_terminology: Terminology preservation
    - ErrorAnalyzer: Aggregate error analysis class
"""

from src.evaluation.correction_metrics import (
    compute_character_error_rate,
    compute_word_error_rate,
    compute_exact_match,
    evaluate_metrics,
)
from src.evaluation.error_analysis import (
    evaluate_edits,
    evaluate_terminology,
    ErrorAnalyzer,
)

__all__ = [
    "compute_character_error_rate",
    "compute_word_error_rate",
    "compute_exact_match",
    "evaluate_metrics",
    "evaluate_edits",
    "evaluate_terminology",
    "ErrorAnalyzer",
]
