from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TextDataset(Dataset):
    """
    Wraps a text file as a stream of overlapping (input_ids, targets) chunks
    for causal language modeling: targets are the input shifted by one token.
    """

    def __init__(
        self,
        text_file,
        tokenizer_file,
        seq_len,
    ):

        self.seq_len = seq_len

        tokenizer = Tokenizer.from_file(
            str(tokenizer_file)
        )

        # Exposed so callers (e.g. pretrain.py) can size the model's
        # output/embedding layer to match this exact tokenizer.
        self.vocab_size = tokenizer.get_vocab_size()

        text = Path(text_file).read_text(
            encoding="utf-8"
        )

        encoding = tokenizer.encode(text)

        token_ids = encoding.ids

        if len(token_ids) <= seq_len:
            raise ValueError(
                f"'{text_file}' only produced {len(token_ids)} tokens, "
                f"which is not enough for seq_len={seq_len}. "
                "Use a smaller --seq-len or a larger text file."
            )

        self.tokens = torch.tensor(
            token_ids,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.tokens) - self.seq_len

    def __getitem__(self, index):

        chunk = self.tokens[
            index:index + self.seq_len + 1
        ]

        input_ids = chunk[:-1]

        targets = chunk[1:]

        return input_ids, targets