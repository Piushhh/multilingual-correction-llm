import torch
import torch.nn as nn

from .config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        self.qkv = nn.Linear(
            config.embedding_dim,
            3 * config.embedding_dim,
        )

        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.context_length,
                    config.context_length,
                    dtype=torch.bool,
                )
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        output = attention_weights @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        return self.output_projection(output)