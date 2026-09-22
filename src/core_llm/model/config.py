from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )