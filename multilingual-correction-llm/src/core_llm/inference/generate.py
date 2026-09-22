"""
Inference interface for the from-scratch Transformer language model.

This is Member 1's "callable interface for text generation and correction
experiments" deliverable. It loads a trained checkpoint + its tokenizer and
exposes:

- a Python class (`LLM`) other team members can import directly
  (e.g. Member 3's correction pipeline, or the FastAPI app), and
- a command-line entry point for quick manual testing.

Example (Python):

    from src.core_llm.inference.generate import LLM

    llm = LLM.from_checkpoint(
        checkpoint_path="checkpoints/base/best.pt",
        tokenizer_path="data/tokenizer/tokenizer.json",
    )
    print(llm.generate("The transformer architecture", max_new_tokens=40))

Example (CLI):

    python -m src.core_llm.inference.generate \
        --checkpoint checkpoints/base/best.pt \
        --tokenizer data/tokenizer/tokenizer.json \
        --prompt "Deep learning is"
"""

import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.core_llm.model.transformer import (
    TransformerConfig,
    CausalTransformerLM,
)


class LLM:
    """Thin wrapper bundling a trained model with its tokenizer for inference."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer_path, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        device = torch.device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        config = TransformerConfig(**checkpoint["config"])
        model = CausalTransformerLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return cls(model, tokenizer, device)

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token="<eos>",
    ):
        """
        Autoregressively generate text following `prompt`.

        - temperature: >1.0 = more random, <1.0 = more confident/greedy,
          0.0 = pure greedy (always pick the highest-probability token).
        - top_k: if set, only sample from the k highest-probability tokens.
        - top_p: if set, use nucleus sampling (sample from the smallest set
          of tokens whose cumulative probability exceeds top_p).
        """

        max_seq_len = self.model.config.max_seq_len

        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor(
            [input_ids], dtype=torch.long, device=self.device
        )

        eos_id = self.tokenizer.token_to_id(eos_token)

        for _ in range(max_new_tokens):
            # The model only has positional embeddings up to max_seq_len,
            # so once the running sequence is longer than that we condition
            # on just the most recent window instead of the whole history.
            conditioning_ids = input_ids[:, -max_seq_len:]

            logits, _ = self.model(conditioning_ids)

            next_token_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token_id = next_token_logits.argmax(
                    dim=-1, keepdim=True
                )
            else:
                next_token_logits = next_token_logits / temperature

                if top_k is not None:
                    top_values, _ = torch.topk(next_token_logits, top_k)
                    threshold = top_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threshold, float("-inf")
                    )

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_mask = cumulative_probs > top_p
                    # Always keep at least the single most likely token.
                    sorted_mask[:, 0] = False

                    sorted_logits = sorted_logits.masked_fill(
                        sorted_mask, float("-inf")
                    )

                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    ).scatter(-1, sorted_indices, sorted_logits)

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if eos_id is not None and next_token_id.item() == eos_id:
                break

        generated_ids = input_ids[0].tolist()
        return self.tokenizer.decode(generated_ids)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json.")
    parser.add_argument("--prompt", required=True, help="Text prompt to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    llm = LLM.from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )

    output = llm.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print(output)
