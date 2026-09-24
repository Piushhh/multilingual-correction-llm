import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TokenPositionalEmbedding
from .transformer_block import TransformerBlock


class TransformerConfig:
    def __init__(
        self,
        vocab_size,
        max_seq_len=256,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout


class CausalTransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embeddings = TokenPositionalEmbedding(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

        self.lm_head.weight = self.embeddings.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, input_ids, targets=None):

        _, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum {self.config.max_seq_len}"
            )

        x = self.embeddings(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss