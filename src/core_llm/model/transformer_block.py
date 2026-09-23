import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .config import ModelConfig


class TransformerBlock(nn.Module):
    """
    Pre-norm decoder Transformer block.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            config.embedding_dim
        )

        self.attention = MultiHeadSelfAttention(
            config
        )

        self.layer_norm_2 = nn.LayerNorm(
            config.embedding_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                config.embedding_dim,
                4 * config.embedding_dim,
            ),
            nn.GELU(),
            nn.Linear(
                4 * config.embedding_dim,
                config.embedding_dim,
            ),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        x = x + self.mlp(
            self.layer_norm_2(x)
        )

        return x