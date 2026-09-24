import difflib
from typing import Any, Dict, List


def detect_changes(original: str, corrected: str) -> List[Dict[str, Any]]:
    """
    Detects changes between the original and corrected text using difflib.
    Returns a structured list of changes.
    """
    changes = []

    orig_words = original.split()
    corr_words = corrected.split()

    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        original_segment = " ".join(orig_words[i1:i2])
        corrected_segment = " ".join(corr_words[j1:j2])

        category = "general_edit"
        if tag == "replace":
            category = "replacement"
        elif tag == "delete":
            category = "deletion"
        elif tag == "insert":
            category = "insertion"

        changes.append({
            "original": original_segment,
            "corrected": corrected_segment,
            "category": category,
            "position": {"start_orig": i1, "end_orig": i2, "start_corr": j1, "end_corr": j2},
            "confidence": 1.0,
        })

    return changes
