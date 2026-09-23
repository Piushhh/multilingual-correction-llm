from pathlib import Path

import sentencepiece as spm


ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "src" / "core_llm" / "tokenizer"

CORPUS_PATH = DATA_DIR / "corpus.txt"
MODEL_PREFIX = OUTPUT_DIR / "tokenizer"

VOCAB_SIZE = 256


def train_tokenizer():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Training corpus not found: {CORPUS_PATH}\n"
            "Create data/raw/corpus.txt first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    spm.SentencePieceTrainer.train(
        input=str(CORPUS_PATH),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
    )

    print("Tokenizer training complete.")
    print(f"Model: {MODEL_PREFIX}.model")
    print(f"Vocabulary: {MODEL_PREFIX}.vocab")


if __name__ == "__main__":
    train_tokenizer()