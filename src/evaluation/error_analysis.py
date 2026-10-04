import json
from src.correction.change_detection import detect_changes

def evaluate_edits(original: str, reference: str, hypothesis: str) -> dict:
    """
    Evaluates edit-level accuracy by comparing:
    - Gold edits: original -> reference
    - Predicted edits: original -> hypothesis
    """
    gold_changes = detect_changes(original, reference)
    pred_changes = detect_changes(original, hypothesis)
    
    # Simple set-based comparison for this baseline
    def normalize_change(c):
        return f"{c['category']}:{c['original']}->{c['corrected']}"
        
    gold_set = {normalize_change(c) for c in gold_changes}
    pred_set = {normalize_change(c) for c in pred_changes}
    
    true_corrections = gold_set.intersection(pred_set)
    missed_errors = gold_set - pred_set
    incorrect_edits = pred_set - gold_set
    
    # Over-correction heuristic: edit on a span that wasn't touched in gold
    # This requires span checking, but for simple tracking:
    over_corrections = len([c for c in incorrect_edits if "->" in c])
    
    return {
        "true_corrections": len(true_corrections),
        "missed_errors": len(missed_errors),
        "incorrect_edits": len(incorrect_edits),
        "over_corrections": over_corrections,
        "precision": len(true_corrections) / max(1, len(pred_set)),
        "recall": len(true_corrections) / max(1, len(gold_set))
    }

def evaluate_terminology(reference: str, hypothesis: str, terminology_glossary: set) -> dict:
    """
    Evaluates whether domain terminology was preserved.
    """
    ref_terms = {word for word in reference.split() if word in terminology_glossary}
    hyp_terms = {word for word in hypothesis.split() if word in terminology_glossary}
    
    preserved = ref_terms.intersection(hyp_terms)
    damaged = ref_terms - hyp_terms
    introduced = hyp_terms - ref_terms
    
    return {
        "terms_preserved": len(preserved),
        "terms_damaged": len(damaged),
        "terms_introduced": len(introduced)
    }

class ErrorAnalyzer:
    def __init__(self, terminology_path=None):
        self.terminology = set()
        if terminology_path:
            # Load glossary if provided
            pass
            
    def analyze_dataset(self, dataset: list) -> dict:
        """
        Analyzes the full dataset and computes aggregate error analysis.
        dataset should contain dicts with 'original', 'reference', 'hypothesis', 'language', 'domain'
        """
        results = {
            "overall": {"cer": 0, "wer": 0, "precision": 0, "recall": 0},
            "by_language": {},
            "by_domain": {}
        }
        
        # Placeholder for full loop implementation
        # In actual execution, this would loop over items and calculate means
        return results
