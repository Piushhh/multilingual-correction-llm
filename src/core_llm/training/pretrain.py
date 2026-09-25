"""
Base pretraining script for CausalTransformerLM.

Usage:
    python -m src.core_llm.training.pretrain

Requirements:
    - data/raw/corpus.txt           : UTF-8 multilingual pretraining corpus
    - src/core_llm/tokenizer/tokenizer.model : trained SentencePiece tokenizer

Output:
    - src/core_llm/checkpoints/tiny_model.pt

Checkpoint format:
    {
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "model_config": dict,         # reconstruct model without guessing
        "tokenizer_path": str,
        "epoch": int,
        "global_step": int,
        "train_loss": float,
        "validation_loss": float,
        "validation_perplexity": float,
        "seed": int,
        "stage": "pretrain",
    }
"""

from pathlib import Path
import math
import torch
from torch.utils.data import random_split

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
    perplexity,
    count_parameters,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "corpus.txt"
TOKENIZER_PATH = REPO_ROOT / "src" / "core_llm" / "tokenizer" / "tokenizer.model"
CHECKPOINT_PATH = REPO_ROOT / "src" / "core_llm" / "checkpoints" / "tiny_model.pt"

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 4
CONTEXT_LENGTH = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 10
TRAIN_SPLIT = 0.9
PATIENCE = 3
MAX_GRAD_NORM = 1.0


def main():
    torch.manual_seed(SEED)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}\n"
            "Run: python -m src.core_llm.scripts.train_tokenizer"
        )
    tokenizer = BPETokenizer(str(TOKENIZER_PATH))
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # ── Corpus ────────────────────────────────────────────────────────────
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(
            f"\nCorpUS NOT FOUND: {CORPUS_PATH}\n"
            "BLOCKED: multilingual pretraining corpus is not available.\n"
            "Supply a UTF-8 text file at data/raw/corpus.txt to begin pretraining.\n"
            "You can build it from OCR output with:\n"
            "  python -m src.core_llm.scripts.build_corpus"
        )

    token_ids = load_corpus_tokens(str(CORPUS_PATH), tokenizer)
    print(f"Total tokens: {len(token_ids):,}")

    if len(token_ids) <= CONTEXT_LENGTH:
        raise ValueError(
            f"Corpus has only {len(token_ids)} tokens but context_length="
            f"{CONTEXT_LENGTH}. Provide a larger corpus."
        )

    # ── Dataset ───────────────────────────────────────────────────────────
    full_dataset = LanguageModelDataset(token_ids, CONTEXT_LENGTH)
    print(f"Dataset size: {len(full_dataset):,} examples")

    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    if val_size == 0:
        raise ValueError("Corpus too small to create a validation split.")

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )
    model = CausalTransformerLM(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Context length:   {CONTEXT_LENGTH}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Learning rate:    {LEARNING_RATE}")
    print(f"Epochs:           {EPOCHS}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            out = model(input_ids, target_ids)
            loss = out["loss"]

            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during training.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1
            global_step += 1

        train_loss = total_train_loss / train_batches if train_batches > 0 else float("inf")
        val_loss = evaluate(model, val_loader, device)
        val_ppl = perplexity(val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Step {global_step} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": config.to_dict(),
                    "tokenizer_path": str(TOKENIZER_PATH),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "validation_perplexity": val_ppl,
                    "seed": SEED,
                    "stage": "pretrain",
                },
                CHECKPOINT_PATH,
            )
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print(f"\nPretraining complete.")
    print(f"Best validation loss:       {best_val_loss:.4f}")
    print(f"Best validation perplexity: {perplexity(best_val_loss):.2f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
