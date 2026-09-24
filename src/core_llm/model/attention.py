import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(
            d_model,
            3 * d_model
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        output = attention_weights @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.d_model
        )

        output = self.output_projection(output)

        return output