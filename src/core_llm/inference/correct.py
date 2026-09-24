import sys
from pathlib import Path
from typing import Optional, Tuple
import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer


CHECKPOINT_PATH = "src/core_llm/checkpoints/correction_model.pt"
TOKENIZER_PATH = "src/core_llm/tokenizer/tokenizer.model"

CONTEXT_LENGTH = 64
MAX_NEW_TOKENS = 30
EOS_TOKEN_ID = 2


def load_model(
    checkpoint_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[CausalTransformerLM, BPETokenizer, torch.device]:
    """
    Load Member 1 custom correction Transformer model, tokenizer, and target device.

    Args:
        checkpoint_path: Path to correction_model.pt (defaults to CHECKPOINT_PATH).
        tokenizer_path: Path to tokenizer.model (defaults to TOKENIZER_PATH).
        device: Torch device (defaults to CUDA if available, else CPU).

    Returns:
        (model, tokenizer, device)
    """
    ckpt_path = Path(checkpoint_path or CHECKPOINT_PATH)
    tok_path = Path(tokenizer_path or TOKENIZER_PATH)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Correction checkpoint not found at: {ckpt_path.resolve()}\n"
            "Ensure the trained model checkpoint exists."
        )

    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found at: {tok_path.resolve()}\n"
            "Ensure tokenizer.model exists."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BPETokenizer(str(tok_path))

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
        str(ckpt_path),
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(
        state_dict,
        strict=False,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def correct_text(
    model: CausalTransformerLM,
    tokenizer: BPETokenizer,
    device: torch.device,
    text: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Run correction inference on the provided text.

    Args:
        model: CausalTransformerLM instance.
        tokenizer: BPETokenizer instance.
        device: Execution device.
        text: Input text to correct.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Corrected text string.
    """
    if not text or not text.strip():
        return ""

    prompt = f"Correct:\n{text.strip()}\nAnswer:\n"
    prompt_ids = tokenizer.encode(prompt)

    # Keep enough room for generated correction within CONTEXT_LENGTH
    if len(prompt_ids) >= CONTEXT_LENGTH:
        prompt_ids = prompt_ids[-(CONTEXT_LENGTH - 1):]

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated_start = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -CONTEXT_LENGTH:]
            output = model(context)
            logits = output["logits"]

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            # Stop if EOS token generated
            if next_token.item() == EOS_TOKEN_ID:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

    generated_tokens = input_ids[0].tolist()[generated_start:]
    decoded = tokenizer.decode(generated_tokens)

    # Clean up newline artifacts if any
    return decoded.strip()


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Deep learnig is a subfeld of machne learning."

    print("Loading correction model...")
    model, tokenizer, device = load_model()

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