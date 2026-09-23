import torch
import torch.nn as nn

from .config import ModelConfig


class LMHead(nn.Module):
    """
    Projects hidden states into vocabulary logits.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.projection = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)