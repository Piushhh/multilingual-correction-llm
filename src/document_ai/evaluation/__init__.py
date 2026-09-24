"""
Evaluation package for Member 2 Document AI.
"""

from src.document_ai.evaluation.ocr_metrics import compute_cer, compute_wer, evaluate_ocr

__all__ = ["compute_cer", "compute_wer", "evaluate_ocr"]
