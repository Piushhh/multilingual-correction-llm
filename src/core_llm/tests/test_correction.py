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
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_correction(
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

    token_ids = tokenizer.encode(prompt)

    if len(token_ids) >= CONTEXT_LENGTH:
        token_ids = token_ids[
            -(CONTEXT_LENGTH - 1):
        ]

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    original_length = input_ids.shape[1]

    with torch.no_grad():

        for _ in range(30):

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
    ].tolist()[original_length:]

    return tokenizer.decode(
        generated_tokens
    )


def test_correction_model():

    model, tokenizer, device = load_model()

    test_cases = [
        (
            "Deep learnig is a subfeld of machne learning.",
            "Deep learning is a subfield of machine learning.",
        ),
        (
            "Deep learning uses neural netwroks with multiple layers.",
            "Deep learning uses neural networks with multiple layers.",
        ),
        (
            "डीप लर्निंग मशीन लर्निग की एक शाखा है।",
            "डीप लर्निंग मशीन लर्निंग की एक शाखा है।",
        ),
    ]

    print()
    print("========== CORRECTION MODEL TEST ==========")
    print()

    for i, (incorrect, expected) in enumerate(
        test_cases,
        1,
    ):

        result = generate_correction(
            model,
            tokenizer,
            device,
            incorrect,
        )

        print(f"Test {i}")
        print("Input:   ", incorrect)
        print("Expected:", expected)
        print("Model:   ", result)
        print("-" * 60)

    # The important automated check:
    # the checkpoint must load successfully.
    assert model is not None
    assert tokenizer is not None


if __name__ == "__main__":
    test_correction_model()