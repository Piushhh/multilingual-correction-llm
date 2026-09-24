"""
Domain terminology database + technical-term / OCR-error detection.

Member 2, Responsibility 5: "Create a verified terminology database and
detect possible technical terms or OCR errors."

The database is a plain JSON file per domain (data/terminology/*.json) so
it's easy for anyone on the team (not just Member 2) to add verified terms
without touching code, and easy to diff/review in a pull request.

Detection uses fuzzy string matching (rapidfuzz) rather than exact lookup,
because the whole point is catching OCR errors: a scanned page might read
"atention" instead of "attention", and exact matching would just silently
miss it. A token that's CLOSE to a verified term but not an exact match is
exactly the "possible technical term or OCR error" the milestone asks for.
"""

import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TERMINOLOGY_DIR = ROOT / "data" / "terminology"

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")  # words of length >= 3, letters/hyphens


class TerminologyDatabase:
    """
    Loads verified terms for one or more domains and checks OCR'd text
    against them.

    JSON file format (data/terminology/<domain>.json):
        {
          "domain": "deep_learning",
          "terms": ["attention", "backpropagation", "gradient", ...]
        }
    """

    def __init__(self, terms_by_domain):
        # Store as {domain: set(lowercased terms)} plus a flat lookup used
        # for fuzzy matching across all loaded domains at once.
        self.terms_by_domain = {
            domain: {t.lower() for t in terms} for domain, terms in terms_by_domain.items()
        }
        self._all_terms = sorted(
            {t for terms in self.terms_by_domain.values() for t in terms}
        )

    @classmethod
    def load(cls, terminology_dir=DEFAULT_TERMINOLOGY_DIR, domains=None):
        terminology_dir = Path(terminology_dir)
        terms_by_domain = {}

        if not terminology_dir.exists():
            return cls(terms_by_domain)

        for json_file in sorted(terminology_dir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            domain = data.get("domain", json_file.stem)
            if domains is not None and domain not in domains:
                continue

            terms_by_domain[domain] = data.get("terms", [])

        return cls(terms_by_domain)

    def is_verified_term(self, word):
        return word.lower() in self._all_terms

    def check_text(
        self,
        text,
        exact_match_skip=True,
        fuzzy_threshold=82,
        max_length_diff=3,
    ):
        """
        Scan `text` for words that are close to (but not exactly) a
        verified term, i.e. likely OCR errors on technical vocabulary, or
        words that exactly match a verified term (flagged separately as
        confirmed technical terms, useful context for the correction
        module downstream).

        Returns a list of findings, each:
            {
                "word": str,           # the word as it appears in the text
                "position": int,       # character offset in `text`
                "status": "verified_term" | "possible_ocr_error",
                "closest_term": str,   # nearest verified term
                "similarity": float,   # 0-100 rapidfuzz score
            }

        fuzzy_threshold: minimum rapidfuzz similarity (0-100) to flag a
        near-miss. 82 is deliberately conservative -- it catches
        single-character OCR substitutions/drops on words of reasonable
        length ("atention"/"attention" scores ~94) without flagging
        unrelated short words that happen to share a few letters.
        """
        if not self._all_terms:
            return []

        findings = []

        for match in _TOKEN_RE.finditer(text):
            word = match.group(0)
            word_lower = word.lower()

            if word_lower in self._all_terms:
                if not exact_match_skip:
                    findings.append(
                        {
                            "word": word,
                            "position": match.start(),
                            "status": "verified_term",
                            "closest_term": word_lower,
                            "similarity": 100.0,
                        }
                    )
                continue

            best = process.extractOne(
                word_lower,
                self._all_terms,
                scorer=fuzz.ratio,
            )

            if best is None:
                continue

            closest_term, similarity, _ = best

            if similarity >= fuzzy_threshold and abs(len(closest_term) - len(word_lower)) <= max_length_diff:
                findings.append(
                    {
                        "word": word,
                        "position": match.start(),
                        "status": "possible_ocr_error",
                        "closest_term": closest_term,
                        "similarity": round(float(similarity), 2),
                    }
                )

        return findings


def bootstrap_terminology_files(output_dir=DEFAULT_TERMINOLOGY_DIR):
    """
    Write a small starter terminology database per domain. This is meant to
    be extended by hand (or by mining a verified corpus) -- it exists so
    the pipeline has something real to run against out of the box, not as
    a finished database.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_terms = {
        "deep_learning": [
            "attention", "backpropagation", "gradient", "transformer",
            "embedding", "tokenizer", "activation", "dropout",
            "convolution", "epoch", "overfitting", "regularization",
            "softmax", "perceptron", "optimizer", "hyperparameter",
        ],
        "computer_science": [
            "algorithm", "recursion", "heap", "stack", "queue",
            "pointer", "compiler", "database", "deadlock", "thread",
            "hashtable", "polymorphism", "inheritance", "bytecode",
        ],
        "mathematics": [
            "derivative", "integral", "eigenvector", "determinant",
            "polynomial", "matrix", "theorem", "inequality",
            "differentiable", "probability", "variance", "logarithm",
        ],
    }

    for domain, terms in seed_terms.items():
        path = output_dir / f"{domain}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"domain": domain, "terms": terms}, f, indent=2, ensure_ascii=False)

    return list(seed_terms.keys())


if __name__ == "__main__":
    written = bootstrap_terminology_files()
    print(f"Wrote starter terminology files for: {written}")
