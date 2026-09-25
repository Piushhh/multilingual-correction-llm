"""
Train the SentencePiece BPE tokenizer.

Input:  data/raw/corpus.txt  (or data/correction/correction_pairs.jsonl as supplement)
Output: src/core_llm/tokenizer/tokenizer.model
        src/core_llm/tokenizer/tokenizer.vocab

Vocabulary size: 256 (matches correction_pairs.jsonl multilingual content)

Usage:
    python -m src.core_llm.scripts.train_tokenizer

The tokenizer is trained with character_coverage=1.0 to handle
all Unicode characters including Devanagari/Hindi script.

Special tokens:
    PAD=0, BOS=1, EOS=2, UNK=3
"""

from pathlib import Path
import tempfile

import sentencepiece as spm


REPO_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_DIR = REPO_ROOT / "src" / "core_llm" / "tokenizer"

CORPUS_PATH = DATA_DIR / "corpus.txt"
CORRECTION_PATH = REPO_ROOT / "data" / "correction" / "correction_pairs.jsonl"
MODEL_PREFIX = OUTPUT_DIR / "tokenizer"

VOCAB_SIZE = 256


def build_training_text(tmp_path: Path) -> Path:
    """
    Combine available text sources for tokenizer training.

    Uses data/raw/corpus.txt if available.
    Supplements with correction pair text if corpus is small or absent.
    """
    import json

    lines = []

    if CORPUS_PATH.is_file():
        lines.extend(CORPUS_PATH.read_text(encoding="utf-8").splitlines())
        print(f"Corpus lines loaded:         {len(lines)}")
    else:
        print(f"WARNING: {CORPUS_PATH} not found. Using correction pairs only.")

    if CORRECTION_PATH.is_file():
        correction_lines = []
        with open(CORRECTION_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    correction_lines.append(record.get("incorrect", ""))
                    correction_lines.append(record.get("correct", ""))
                except json.JSONDecodeError:
                    continue
        lines.extend(correction_lines)
        print(f"Correction pair lines added: {len(correction_lines)}")

    if not lines:
        raise ValueError(
            "No training text available.\n"
            "Supply data/raw/corpus.txt or data/correction/correction_pairs.jsonl."
        )

    combined = tmp_path / "combined_corpus.txt"
    combined.write_text("\n".join(lines), encoding="utf-8")
    return combined


def train_tokenizer(input_path: Path = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        if input_path is None:
            input_path = build_training_text(tmp_path)

        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(MODEL_PREFIX),
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            unk_id=3,
        )

    print(f"\nTokenizer training complete.")
    print(f"Model:      {MODEL_PREFIX}.model")
    print(f"Vocab file: {MODEL_PREFIX}.vocab")
    print(f"Vocab size: {VOCAB_SIZE}")


if __name__ == "__main__":
    train_tokenizer()
