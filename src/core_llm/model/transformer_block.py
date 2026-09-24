import torch.nn as nn

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.0,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x