from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ModelConfig:
    """
    Configuration for the CausalTransformerLM.

    All training stages (pretraining, domain adaptation, correction fine-tuning)
    must store and reload this config via checkpoint metadata so that the model
    architecture can be reconstructed exactly.
    """

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
                f"embedding_dim ({self.embedding_dim}) must be "
                f"divisible by num_heads ({self.num_heads})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dict suitable for checkpoint storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """Reconstruct a ModelConfig from a plain dict (e.g. from a checkpoint)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
