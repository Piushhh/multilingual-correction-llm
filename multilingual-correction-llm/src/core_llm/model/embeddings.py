import torch
import torch.nn as nn


class TokenPositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        dropout=0.0,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        return self.dropout(x)