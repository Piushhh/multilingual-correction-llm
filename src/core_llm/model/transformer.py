"""
CausalTransformerLM — decoder-only Transformer language model.

Architecture:
    TokenPositionEmbedding
        → N × TransformerBlock (pre-norm, causal attention + GELU MLP)
            → final LayerNorm
                → LMHead (linear projection to vocab_size logits)

Training:
    Forward with targets → cross-entropy loss (ignore_index=-100 supported)

Inference:
    model.generate(input_ids, max_new_tokens, temperature, top_k)
    Returns extended input_ids tensor. Extract only new tokens from
    [:, original_length:] if you need only the generated portion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import TokenPositionEmbedding
from .lm_head import LMHead
from .transformer_block import TransformerBlock


class CausalTransformerLM(nn.Module):
    """
    Compact decoder-only Transformer language model trained from scratch.

    No external pretrained weights. Fully compatible with CPU and CUDA.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embeddings = TokenPositionEmbedding(config)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)

        self.lm_head = LMHead(config)

        self.apply(self._initialize_weights)

    # ── Weight initialization ─────────────────────────────────────────────

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Parameter count ───────────────────────────────────────────────────

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input_ids: [batch_size, sequence_length] — token ids
            targets:   [batch_size, sequence_length] — optional target ids
                       Use -100 at positions to ignore in loss calculation.

        Returns:
            dict with keys:
                "logits": [batch_size, sequence_length, vocab_size]
                "loss":   scalar tensor if targets provided, else None
        """
        x = self.embeddings(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    # ── Autoregressive generation ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive token generation.

        Args:
            input_ids:      [1, prompt_length] starting token ids
            max_new_tokens: how many new tokens to generate
            temperature:    sampling temperature (1.0 = no scaling)
            top_k:          top-k filtering; None = no filtering
            eos_token_id:   if set, stops generation when EOS is produced

        Returns:
            [1, prompt_length + generated_length] token ids
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate to context window
            input_ids_cond = input_ids[:, -self.config.context_length:]

            outputs = self(input_ids_cond)
            logits = outputs["logits"][:, -1, :]  # [batch, vocab]

            # Temperature scaling
            if temperature > 0:
                logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                min_value = values[:, [-1]]
                logits = torch.where(
                    logits < min_value,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            if temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids
