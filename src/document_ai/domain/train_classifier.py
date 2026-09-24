"""
Domain classifier training / keyword compilation utility.
Extracts domain-specific keywords and term frequencies from domain corpora.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Set
import re


def extract_keywords_from_corpus(
    corpus_path: Path,
    min_freq: int = 2,
    top_k: int = 50,
) -> List[str]:
    """
    Extract high-frequency candidate domain keywords from a raw text corpus.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    text = path.read_text(encoding="utf-8").lower()
    words = re.findall(r"[\w\u0900-\u097F\-]+", text)
    counts = Counter(words)

    # Filter out single character tokens
    filtered = {w: c for w, c in counts.items() if len(w) > 2 and c >= min_freq}
    return [w for w, _ in Counter(filtered).most_common(top_k)]
