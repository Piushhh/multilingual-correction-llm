import torch
import torch.nn as nn

from .config import ModelConfig


class TokenPositionEmbedding(nn.Module):
    """
    Learned token + learned position embeddings.

    Supports sequences up to config.context_length.
    Raises ValueError if sequence_length exceeds context_length.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.position_embedding.num_embeddings}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)

        embeddings = token_embeddings + position_embeddings

        return self.dropout(embeddings)
