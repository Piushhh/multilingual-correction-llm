
from pathlib import Path
import math

import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer
from src.core_llm.training.dataset import (
    LanguageModelDataset,
    load_corpus_tokens,
    create_dataloader,
)
from src.core_llm.training.train_utils import (
    train_one_epoch,
    evaluate,
)


DOMAIN_CORPUS_PATH = "data/raw/clean/domain_corpus.txt"

BASE_CHECKPOINT_PATH = (
    "src/core_llm/checkpoints/tiny_model.pt"
)

DOMAIN_CHECKPOINT_PATH = (
    "src/core_llm/checkpoints/"
    "domain_adapted_model.pt"
)

TOKENIZER_PATH = (
    "src/core_llm/tokenizer/tokenizer.model"
)

BATCH_SIZE = 4
CONTEXT_LENGTH = 32
LEARNING_RATE = 1e-4
EPOCHS = 5

VALIDATION_RATIO = 0.1
PATIENCE = 2


def main():

    # -------------------------
    # Device
    # -------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # -------------------------
    # Tokenizer
    # -------------------------

    tokenizer = BPETokenizer(
        TOKENIZER_PATH
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    # -------------------------
    # Load domain corpus
    # -------------------------

    token_ids = load_corpus_tokens(
        DOMAIN_CORPUS_PATH,
        tokenizer,
    )

    print(
        "Total domain tokens:",
        len(token_ids),
    )

    # -------------------------
    # Train / validation split
    # -------------------------

    split_index = int(
        len(token_ids)
        * (1 - VALIDATION_RATIO)
    )

    train_tokens = token_ids[:split_index]
    validation_tokens = token_ids[split_index:]

    train_dataset = LanguageModelDataset(
        train_tokens,
        CONTEXT_LENGTH,
    )

    validation_dataset = LanguageModelDataset(
        validation_tokens,
        CONTEXT_LENGTH,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(validation_dataset),
    )

    # -------------------------
    # Model configuration
    # -------------------------

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )

    model = CausalTransformerLM(
        config
    ).to(device)

    # -------------------------
    # Load base model
    # -------------------------

    checkpoint = torch.load(
        BASE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded base checkpoint:",
        BASE_CHECKPOINT_PATH,
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------
    # Best model tracking
    # -------------------------

    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    # -------------------------
    # Domain adaptation
    # -------------------------

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_loss = evaluate(
            model,
            validation_loader,
            device,
        )

        # -------------------------
        # Validation perplexity
        # -------------------------

        validation_perplexity = math.exp(
            min(validation_loss, 20)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Validation Loss: "
            f"{validation_loss:.4f} "
            f"- Validation Perplexity: "
            f"{validation_perplexity:.4f}"
        )

        # -------------------------
        # Save best checkpoint
        # -------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            patience_counter = 0

            Path(
                DOMAIN_CHECKPOINT_PATH
            ).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "config":
                        config,

                    "tokenizer_path":
                        TOKENIZER_PATH,

                    "base_checkpoint":
                        BASE_CHECKPOINT_PATH,

                    "validation_loss":
                        validation_loss,

                    "validation_perplexity":
                        validation_perplexity,

                    "epoch":
                        best_epoch,
                },
                DOMAIN_CHECKPOINT_PATH,
            )

            print(
                f"  ✓ Best domain model saved "
                f"(validation loss: "
                f"{best_validation_loss:.4f})"
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        # -------------------------
        # Early stopping
        # -------------------------

        if patience_counter >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # -------------------------
    # Final information
    # -------------------------

    print()
    print(
        "Domain adaptation complete."
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        f"{best_validation_loss:.4f}",
    )

    print(
        "Best validation perplexity:",
        f"{math.exp(min(best_validation_loss, 20)):.4f}",
    )

    print()
    print(
        "Best domain-adapted checkpoint saved:"
    )

    print(DOMAIN_CHECKPOINT_PATH)


if __name__ == "__main__":
    main()