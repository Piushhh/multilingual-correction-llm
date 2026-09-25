import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .config import ModelConfig


class TransformerBlock(nn.Module):
    """
    Pre-norm decoder Transformer block.

    Architecture per token position:
        x = x + Attention(LayerNorm(x))    # residual + attention
        x = x + MLP(LayerNorm(x))          # residual + feed-forward

    MLP uses GELU activation and 4× expansion factor.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadSelfAttention(config)

        self.layer_norm_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attention(self.layer_norm_1(x))
        # Pre-norm MLP with residual
        x = x + self.mlp(self.layer_norm_2(x))
        return x
