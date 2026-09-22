"""
Evaluation for Member 2's components.

Member 2 Milestone 6: "OCR evaluation." + general "Evaluation and error
analysis" from the learning roadmap.

Two things are evaluated here:
1. OCR quality: Character Error Rate (CER) and Word Error Rate (WER)
   against ground-truth transcriptions, using Levenshtein edit distance.
2. Language detection accuracy against labeled examples.

Domain classifier evaluation already happens inline in
train_classifier.py (sklearn's classification_report on a held-out split)
-- not duplicated here.
"""

from src.document_ai.language.detector import detect_language


def _levenshtein(a, b):
    """Standard O(len(a)*len(b)) edit-distance DP. Fine at the
    line/word-list lengths this is used on; not meant for huge documents."""
    if len(a) < len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            insert_cost = previous_row[j] + 1
            delete_cost = current_row[j - 1] + 1
            substitute_cost = previous_row[j - 1] + (char_a != char_b)
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
        previous_row = current_row

    return previous_row[-1]


def character_error_rate(hypothesis, reference):
    """CER = edit_distance(hyp, ref) / len(ref), at the character level."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return _levenshtein(hypothesis, reference) / len(reference)


def word_error_rate(hypothesis, reference):
    """WER = edit_distance(hyp_words, ref_words) / len(ref_words)."""
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _levenshtein(hyp_words, ref_words) / len(ref_words)


def evaluate_ocr(pairs):
    """
    pairs: list of (hypothesis_text, reference_text) tuples.

    Returns {"mean_cer": float, "mean_wer": float, "n": int}. Mean, not
    pooled/weighted, so a handful of long documents can't drown out
    performance on the short ones -- both matter equally for a document
    correction tool.
    """
    if not pairs:
        return {"mean_cer": 0.0, "mean_wer": 0.0, "n": 0}

    cers = [character_error_rate(hyp, ref) for hyp, ref in pairs]
    wers = [word_error_rate(hyp, ref) for hyp, ref in pairs]

    return {
        "mean_cer": round(sum(cers) / len(cers), 4),
        "mean_wer": round(sum(wers) / len(wers), 4),
        "n": len(pairs),
    }


def evaluate_language_detection(labeled_examples):
    """
    labeled_examples: list of (text, true_language) tuples where
    true_language is one of "en", "hi", "code_mixed".

    Returns {"accuracy": float, "n": int, "confusion": {(true, pred): count}}.
    """
    if not labeled_examples:
        return {"accuracy": 0.0, "n": 0, "confusion": {}}

    correct = 0
    confusion = {}

    for text, true_language in labeled_examples:
        predicted = detect_language(text)["language"]
        confusion[(true_language, predicted)] = confusion.get((true_language, predicted), 0) + 1
        if predicted == true_language:
            correct += 1

    return {
        "accuracy": round(correct / len(labeled_examples), 4),
        "n": len(labeled_examples),
        "confusion": confusion,
    }
