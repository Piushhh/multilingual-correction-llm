"""
Build the pretraining corpus from OCR output.

Input:  data/raw/ocr_results_easyocr_clean.txt
Output: data/raw/corpus.txt

Usage:
    python -m src.core_llm.scripts.build_corpus

The script extracts lines beginning with "Text :" from the OCR results,
strips the prefix, filters out empty lines and "None" values, and
writes one text sample per line to corpus.txt.

If the OCR file does not exist, the script exits with a clear error.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
OCR_FILE = REPO_ROOT / "data" / "raw" / "ocr_results_easyocr_clean.txt"
CORPUS_FILE = REPO_ROOT / "data" / "raw" / "corpus.txt"


def build_corpus():
    if not OCR_FILE.exists():
        raise FileNotFoundError(
            f"\nOCR source file not found: {OCR_FILE}\n"
            "BLOCKED: Supply data/raw/ocr_results_easyocr_clean.txt\n"
            "to build the pretraining corpus."
        )

    lines = OCR_FILE.read_text(encoding="utf-8").splitlines()
    texts = []

    for line in lines:
        if not line.startswith("Text :"):
            continue
        text = line.replace("Text :", "", 1).strip()
        if not text or text.lower() == "none":
            continue
        texts.append(text)

    if not texts:
        raise ValueError(
            f"No valid text lines found in {OCR_FILE}. "
            "Check that the file contains lines starting with 'Text :'."
        )

    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_FILE.write_text("\n".join(texts), encoding="utf-8")

    print(f"Corpus created:         {CORPUS_FILE}")
    print(f"Number of text samples: {len(texts)}")
    print(f"Total characters:       {sum(len(t) for t in texts):,}")


if __name__ == "__main__":
    build_corpus()
