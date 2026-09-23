import sys
import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer


CHECKPOINT_PATH = (
    "src/core_llm/checkpoints/correction_model.pt"
)

TOKENIZER_PATH = (
    "src/core_llm/tokenizer/tokenizer.model"
)

CONTEXT_LENGTH = 64
MAX_NEW_TOKENS = 30


def load_model():

    device = torch.device("cpu")

    tokenizer = BPETokenizer(
        TOKENIZER_PATH
    )

    config = ModelConfig(
        vocab_size=256,
        context_length=CONTEXT_LENGTH,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )

    model = CausalTransformerLM(config)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def correct_text(
    model,
    tokenizer,
    device,
    text,
):

    prompt = (
        "Correct:\n"
        + text
        + "\nAnswer:\n"
    )

    prompt_ids = tokenizer.encode(prompt)

    # Keep enough room for generated correction.
    if len(prompt_ids) >= CONTEXT_LENGTH:
        prompt_ids = prompt_ids[
            -(CONTEXT_LENGTH - 1):
        ]

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated_start = input_ids.shape[1]

    with torch.no_grad():

        for _ in range(MAX_NEW_TOKENS):

            context = input_ids[
                :, -CONTEXT_LENGTH:
            ]

            output = model(context)

            logits = output["logits"]

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

    generated_tokens = input_ids[
        0
    ].tolist()[generated_start:]

    return tokenizer.decode(
        generated_tokens
    )


def main():

    if len(sys.argv) > 1:

        text = " ".join(
            sys.argv[1:]
        )

    else:

        text = (
            "Deep learnig is a subfeld "
            "of machne learning."
        )

    print(
        "Loading correction model..."
    )

    model, tokenizer, device = (
        load_model()
    )

    print("Input:")
    print(text)

    corrected = correct_text(
        model,
        tokenizer,
        device,
        text,
    )

    print("\nModel correction:")
    print(corrected)


if __name__ == "__main__":
    main()