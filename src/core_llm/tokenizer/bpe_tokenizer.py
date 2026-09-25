"""
BPE Tokenizer wrapper around SentencePiece.

Supports English, Hindi/Devanagari, and code-mixed text.

Special token IDs (matches SentencePiece training defaults):
    PAD  = 0
    BOS  = 1
    EOS  = 2
    UNK  = 3
"""

from pathlib import Path

import sentencepiece as spm


class BPETokenizer:
    """
    SentencePiece BPE tokenizer for multilingual text.

    Args:
        model_path: Path to the .model file produced by SentencePiece training.

    Special tokens:
        PAD_ID = 0, BOS_ID = 1, EOS_ID = 2, UNK_ID = 3
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, model_path: str):
        model_path = str(model_path)
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}\n"
                "Run src/core_llm/scripts/train_tokenizer.py to create it."
            )

        self.model_path = model_path
        self.processor = spm.SentencePieceProcessor(model_file=model_path)

    # ── Core properties ───────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self.processor.get_piece_size()

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of integer token IDs.

        Args:
            text: UTF-8 string (English, Hindi, or code-mixed).

        Returns:
            List of integer token IDs. Returns [] for empty input.
        """
        if not text:
            return []
        return self.processor.encode(text, out_type=int)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a list of strings."""
        return [self.encode(t) for t in texts]

    # ── Decoding ──────────────────────────────────────────────────────────

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of integer token IDs back to a string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded UTF-8 string.
        """
        if not token_ids:
            return ""
        return self.processor.decode(token_ids)

    def decode_batch(self, batch_token_ids: list[list[int]]) -> list[str]:
        """Decode a list of token ID sequences."""
        return [self.decode(ids) for ids in batch_token_ids]

    # ── Utilities ─────────────────────────────────────────────────────────

    def id_to_piece(self, token_id: int) -> str:
        """Return the string piece for a token ID."""
        return self.processor.id_to_piece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """Return the token ID for a string piece."""
        return self.processor.piece_to_id(piece)
