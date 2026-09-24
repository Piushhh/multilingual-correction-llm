import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# tokenizer.py -> tokenizer/ -> core_llm/ -> src/ -> project root
ROOT = Path(__file__).resolve().parents[3]

DEFAULT_TRAIN_FILES = [
    ROOT / "data" / "raw" / "train.txt",
    ROOT / "data" / "domain" / "train.txt",
]
TOKENIZER_DIR = ROOT / "data" / "tokenizer"
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"


def train_tokenizer(
    train_files=None,
    vocab_size=8000,
    min_frequency=2,
    output_file=TOKENIZER_FILE,
):
    """
    Train a byte-level BPE tokenizer over one or more text files and save it.

    A byte-level pre-tokenizer is used (rather than a whitespace-based one)
    specifically so the tokenizer can represent English, Hindi (Devanagari
    script) and code-mixed text without any <unk> fallback: every possible
    input byte sequence is representable.
    """

    train_files = [Path(f) for f in (train_files or DEFAULT_TRAIN_FILES)]
    train_files = [f for f in train_files if f.exists() and f.stat().st_size > 0]

    if not train_files:
        raise FileNotFoundError(
            "No non-empty training files found. Checked: "
            + ", ".join(str(f) for f in (train_files or DEFAULT_TRAIN_FILES))
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>"
        )
    )

    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.train(
        files=[str(f) for f in train_files],
        trainer=trainer,
    )

    tokenizer.save(str(output_file))

    print("Tokenizer trained successfully.")
    print(f"Trained on: {[str(f) for f in train_files]}")
    print(f"Saved to: {output_file}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Target vocabulary size (keep small for small datasets).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency to merge.",
    )
    parser.add_argument(
        "--output",
        default=str(TOKENIZER_FILE),
        help="Where to save tokenizer.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_file=args.output,
    )