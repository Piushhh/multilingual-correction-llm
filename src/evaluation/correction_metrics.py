import Levenshtein

def compute_character_error_rate(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate (CER)."""
    if not reference:
        return float(len(hypothesis))
    return Levenshtein.distance(reference, hypothesis) / len(reference)

def compute_word_error_rate(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return float(len(hyp_words))
        
    # Levenshtein can work on lists of words to compute WER
    # But for a simple implementation we can use distance on space-separated tokens
    # Better approach is to use standard WER calculation:
    d = Levenshtein.distance(ref_words, hyp_words)
    return d / len(ref_words)

def compute_exact_match(reference: str, hypothesis: str) -> float:
    """Computes exact match score."""
    return 1.0 if reference.strip() == hypothesis.strip() else 0.0

def evaluate_metrics(reference: str, hypothesis: str) -> dict:
    """Computes all basic correction metrics."""
    return {
        "cer": compute_character_error_rate(reference, hypothesis),
        "wer": compute_word_error_rate(reference, hypothesis),
        "exact_match": compute_exact_match(reference, hypothesis)
    }
