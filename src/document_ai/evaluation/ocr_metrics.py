"""
OCR evaluation metrics: Character Error Rate (CER) and Word Error Rate (WER).
Computes exact edit-distance-based error rates between reference text and OCR hypothesis.
"""

from typing import Dict, List, Sequence, Union

try:
    import Levenshtein
    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False


def _edit_distance(seq1: Sequence, seq2: Sequence) -> int:
    """Fallback pure-Python Wagner-Fischer edit distance."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) = Levenshtein_distance(ref, hyp) / len(ref).

    Args:
        reference: Ground-truth reference string.
        hypothesis: Predicted OCR string.

    Returns:
        Float CER value (0.0 = perfect match).
    """
    ref = reference.strip()
    hyp = hypothesis.strip()

    if not ref:
        return 1.0 if hyp else 0.0

    if _HAS_LEVENSHTEIN:
        dist = Levenshtein.distance(ref, hyp)
    else:
        dist = _edit_distance(list(ref), list(hyp))

    return round(dist / len(ref), 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) = Word_edit_distance(ref, hyp) / num_words(ref).

    Args:
        reference: Ground-truth reference string.
        hypothesis: Predicted OCR string.

    Returns:
        Float WER value (0.0 = perfect match).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    if _HAS_LEVENSHTEIN:
        dist = Levenshtein.distance(ref_words, hyp_words)
    else:
        dist = _edit_distance(ref_words, hyp_words)

    return round(dist / len(ref_words), 4)


def evaluate_ocr(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Evaluate OCR output against reference text.

    Returns:
        {"cer": float, "wer": float, "exact_match": float}
    """
    cer = compute_cer(reference, hypothesis)
    wer = compute_wer(reference, hypothesis)
    em = 1.0 if reference.strip() == hypothesis.strip() else 0.0
    return {
        "cer": cer,
        "wer": wer,
        "exact_match": em,
    }
