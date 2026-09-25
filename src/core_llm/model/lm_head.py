import torch
import torch.nn as nn

from .config import ModelConfig


class LMHead(nn.Module):
    """
    Projects hidden states into vocabulary logits.

    No bias, no softmax — raw logits are returned.
    Cross-entropy loss is computed externally.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.projection = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length, embedding_dim]

        Returns:
            logits: [batch_size, sequence_length, vocab_size]
        """
        return self.projection(x)
