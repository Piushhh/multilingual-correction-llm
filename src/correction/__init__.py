"""
src.correction — Correction model package (Member 3).

Public exports:
    - CorrectionEngine: Main inference engine
    - PromptFormatter: Prompt building utilities
    - detect_changes: Word-level change detection
"""

from src.correction.inference import CorrectionEngine
from src.correction.prompts import PromptFormatter
from src.correction.change_detection import detect_changes

__all__ = ["CorrectionEngine", "PromptFormatter", "detect_changes"]
