"""
Dataset utilities for causal language model training.

LanguageModelDataset:
    Sliding-window dataset from a flat token-id list.
    Each sample is (input_ids, target_ids) where target is shifted by 1.

load_corpus_tokens:
    Reads a UTF-8 text file and encodes it with a tokenizer.

create_dataloader:
    Wraps a Dataset in a DataLoader with sensible defaults.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class LanguageModelDataset(Dataset):
    """
    Sliding-window next-token-prediction dataset.

    Each item at index i returns:
        input_ids:  token_ids[i : i + context_length]
        target_ids: token_ids[i+1 : i + context_length + 1]

    This correctly implements causal next-token prediction.

    Args:
        token_ids:      Flat list of integer token IDs.
        context_length: Length of each training window.
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
    ):
        if len(token_ids) <= context_length:
            raise ValueError(
                f"Corpus has {len(token_ids)} tokens but context_length is "
                f"{context_length}. Need at least {context_length + 1} tokens."
            )

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        # Each sample needs context_length + 1 consecutive tokens
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index: int):
        input_ids = self.token_ids[index : index + self.context_length]
        target_ids = self.token_ids[index + 1 : index + self.context_length + 1]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def load_corpus_tokens(
    corpus_path: str,
    tokenizer,
    encoding: str = "utf-8",
) -> list[int]:
    """
    Read a UTF-8 text file and tokenize the entire content.

    Args:
        corpus_path: Path to a plain-text corpus file.
        tokenizer:   BPETokenizer instance.
        encoding:    File encoding (default utf-8).

    Returns:
        Flat list of integer token IDs.

    Raises:
        FileNotFoundError: If the corpus file does not exist.
        ValueError:        If the file is empty after reading.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    path = Path(corpus_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus file not found: {path.resolve()}\n"
            "Provide a plain-text corpus to begin training."
        )

    text = path.read_text(encoding=encoding)

    if not text.strip():
        raise ValueError(f"Corpus file is empty: {path.resolve()}")

    token_ids = tokenizer.encode(text)

    if not token_ids:
        raise ValueError(
            f"Tokenizer produced 0 tokens for corpus: {path.resolve()}"
        )

    return token_ids


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from a dataset.

    Args:
        dataset:     PyTorch Dataset.
        batch_size:  Batch size.
        shuffle:     Shuffle each epoch.
        num_workers: DataLoader worker processes (0 = main process).

    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
