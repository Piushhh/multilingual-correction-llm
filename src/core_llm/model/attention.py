import torch
import torch.nn as nn

from .config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with pre-registered causal mask.

    Q, K, V are computed via a single fused projection then split.
    The causal mask is registered as a buffer so it moves with the model
    to any device automatically.

    Future tokens CANNOT affect earlier token predictions because the mask
    fills -inf into all upper-triangular attention scores before softmax.
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

        # Fused QKV projection
        self.qkv = nn.Linear(
            config.embedding_dim,
            3 * config.embedding_dim,
            bias=True,
        )

        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Register causal mask as a non-parameter buffer.
        # Shape: [context_length, context_length]
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
        """
        Args:
            x: [batch_size, sequence_length, embedding_dim]

        Returns:
            [batch_size, sequence_length, embedding_dim]
        """
        batch_size, sequence_length, _ = x.shape

        # ── Q / K / V projections ────────────────────────────────────
        qkv = self.qkv(x)  # [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, T, D]

        # ── Reshape to [B, heads, T, head_dim] ──────────────────────
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # ── Scaled dot-product attention ─────────────────────────────
        # [B, heads, T, T]
        attention_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask: mask out future positions with -inf
        mask = self.causal_mask[:sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # ── Weighted sum of values ────────────────────────────────────
        # [B, heads, T, head_dim] -> [B, T, D]
        output = attention_weights @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, sequence_length, self.embedding_dim)

        return self.output_projection(output)
