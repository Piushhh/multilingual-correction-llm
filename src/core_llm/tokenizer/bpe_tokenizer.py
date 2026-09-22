from pathlib import Path

import sentencepiece as spm


class BPETokenizer:
    def __init__(self, model_path: str):
        self.model_path = str(model_path)

        self.processor = spm.SentencePieceProcessor(
            model_file=self.model_path
        )

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(
            text,
            out_type=int,
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.processor.decode(token_ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [
            self.encode(text)
            for text in texts
        ]

    def decode_batch(
        self,
        batch_token_ids: list[list[int]],
    ) -> list[str]:
        return [
            self.decode(token_ids)
            for token_ids in batch_token_ids
        ]