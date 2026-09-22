
from pathlib import Path

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer
from src.core_llm.training.dataset import (
    LanguageModelDataset,
    load_corpus_tokens,
    create_dataloader,
)


CORPUS_PATH = "data/raw/corpus.txt"
TOKENIZER_PATH = "src/core_llm/tokenizer/tokenizer.model"

CHECKPOINT_PATH = "src/core_llm/checkpoints/tiny_model.pt"

BATCH_SIZE = 4
CONTEXT_LENGTH = 32
LEARNING_RATE = 3e-4
EPOCHS = 10

TRAIN_SPLIT = 0.9
PATIENCE = 2


def evaluate(model, dataloader, device, vocab_size):
    """
    Evaluate the model on validation data.
    Returns average validation loss.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for input_ids, target_ids in dataloader:

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            output = model(input_ids)

            logits = output["logits"]

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
            )

            total_loss += loss.item()
            total_batches += 1

    model.train()

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


def main():

    # -------------------------
    # Device
    # -------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -------------------------
    # Tokenizer
    # -------------------------

    tokenizer = BPETokenizer(TOKENIZER_PATH)

    print("Vocabulary size:", tokenizer.vocab_size)

    # -------------------------
    # Load corpus
    # -------------------------

    token_ids = load_corpus_tokens(
        CORPUS_PATH,
        tokenizer,
    )

    print("Total tokens:", len(token_ids))

    # -------------------------
    # Dataset
    # -------------------------

    full_dataset = LanguageModelDataset(
        token_ids,
        CONTEXT_LENGTH,
    )

    print("Full dataset size:", len(full_dataset))

    # -------------------------
    # Train / Validation split
    # -------------------------

    train_size = int(
        TRAIN_SPLIT * len(full_dataset)
    )

    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # -------------------------
    # Model
    # -------------------------

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )

    model = CausalTransformerLM(config).to(device)

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

    best_val_loss = float("inf")
    patience_counter = 0

    # -------------------------
    # Training
    # -------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0
        train_batches = 0

        for input_ids, target_ids in train_dataloader:

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            output = model(input_ids)

            logits = output["logits"]

            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                target_ids.reshape(-1),
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        train_loss = (
            total_train_loss / train_batches
            if train_batches > 0
            else float("inf")
        )

        # -------------------------
        # Validation
        # -------------------------

        val_loss = evaluate(
            model,
            val_dataloader,
            device,
            tokenizer.vocab_size,
        )

        # -------------------------
        # Perplexity
        # -------------------------

        val_perplexity = math.exp(
            min(val_loss, 20)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val Perplexity: {val_perplexity:.4f}"
        )

        # -------------------------
        # Save best checkpoint
        # -------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            Path(CHECKPOINT_PATH).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "tokenizer_path": TOKENIZER_PATH,
                    "best_val_loss": best_val_loss,
                    "val_perplexity": val_perplexity,
                    "epoch": epoch + 1,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"  ✓ Best model saved "
                f"(validation loss: {best_val_loss:.4f})"
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
    print("Training complete.")
    print("Best validation loss:", f"{best_val_loss:.4f}")
    print(
        "Best validation perplexity:",
        f"{math.exp(min(best_val_loss, 20)):.4f}",
    )
    print()
    print("Best model checkpoint saved:")
    print(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
