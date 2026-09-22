import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import TokenPositionEmbedding
from .lm_head import LMHead
from .transformer_block import TransformerBlock


class CausalTransformerLM(nn.Module):
    """
    Compact decoder-only Transformer language model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embeddings = TokenPositionEmbedding(config)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.lm_head = LMHead(config)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
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
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        self.eval()

        for _ in range(max_new_tokens):

            input_ids_cond = input_ids[
                :, -self.config.context_length:
            ]

            outputs = self(input_ids_cond)

            logits = outputs["logits"][:, -1, :]

            logits = logits / max(
                temperature,
                1e-8,
            )

            if top_k is not None:
                top_k_value = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    top_k_value,
                )

                minimum_value = values[:, [-1]]

                logits = torch.where(
                    logits < minimum_value,
                    torch.full_like(
                        logits,
                        float("-inf"),
                    ),
                    logits,
                )

            probabilities = F.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

        return input_ids