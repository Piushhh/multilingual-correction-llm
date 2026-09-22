from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]

OCR_FILE = ROOT_DIR / "data" / "raw" / "ocr_results_easyocr_clean.txt"
CORPUS_FILE = ROOT_DIR / "data" / "raw" / "corpus.txt"


def build_corpus():
    if not OCR_FILE.exists():
        raise FileNotFoundError(f"OCR file not found: {OCR_FILE}")

    texts = []

    lines = OCR_FILE.read_text(encoding="utf-8").splitlines()

    for line in lines:
        if not line.startswith("Text :"):
            continue

        text = line.replace("Text :", "", 1).strip()

        if not text:
            continue

        if text.lower() == "none":
            continue

        texts.append(text)

    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)

    CORPUS_FILE.write_text(
        "\n".join(texts),
        encoding="utf-8",
    )

    print(f"Corpus created: {CORPUS_FILE}")
    print(f"Number of text samples: {len(texts)}")


if __name__ == "__main__":
    build_corpus()