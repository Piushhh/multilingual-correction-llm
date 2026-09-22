from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
    ):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index: int):
        input_ids = self.token_ids[
            index:index + self.context_length
        ]

        target_ids = self.token_ids[
            index + 1:index + self.context_length + 1
        ]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def load_corpus_tokens(
    corpus_path: str,
    tokenizer,
) -> list[int]:
    text = Path(corpus_path).read_text(
        encoding="utf-8"
    )

    return tokenizer.encode(text)


def create_dataloader(
    dataset,
    batch_size: int = 4,
    shuffle: bool = True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )