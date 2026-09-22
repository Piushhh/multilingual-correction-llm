import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer
import sys


TOKENIZER_PATH = (
    "src/core_llm/tokenizer/tokenizer.model"
)

BASE_CHECKPOINT_PATH = (
    "src/core_llm/checkpoints/tiny_model.pt"
)

DOMAIN_CHECKPOINT_PATH = (
    "src/core_llm/checkpoints/"
    "domain_adapted_model.pt"
)

CONTEXT_LENGTH = 32


def load_model(
    checkpoint_path=DOMAIN_CHECKPOINT_PATH,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = BPETokenizer(
        TOKENIZER_PATH
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )

    model = CausalTransformerLM(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 30,
):
    token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the most recent context
            input_context = input_ids[
                :, -CONTEXT_LENGTH:
            ]

            # Forward pass
            output = model(input_context)

            logits = output["logits"]

            # Get logits for the next token
            next_token_logits = (
                logits[:, -1, :]
            )

            # Greedy decoding:
            # select the token with the highest probability
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            # Add the predicted token
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

    return tokenizer.decode(
        input_ids[0].tolist()
    )


def main():

    model, tokenizer, device = load_model()

    prompt =" ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Deep Learning"

    generated = generate_text(
        model,
        tokenizer,
        device,
        prompt,
        max_new_tokens=30,
    )

    print()
    print("Checkpoint:")
    print(DOMAIN_CHECKPOINT_PATH)

    print()
    print("Prompt:", prompt)

    print()
    print("Generated:", generated)


if __name__ == "__main__":
    main()