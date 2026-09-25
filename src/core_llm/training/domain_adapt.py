"""
Domain-adaptive pretraining script.

Continues training from a base pretrained checkpoint on a domain-specific corpus.
Domain: deep_learning (configurable via DOMAIN constant)

Usage:
    python -m src.core_llm.training.domain_adapt

Requirements:
    - src/core_llm/checkpoints/tiny_model.pt        : base checkpoint
    - data/raw/clean/domain_corpus.txt              : domain text corpus
    - src/core_llm/tokenizer/tokenizer.model        : tokenizer

Output:
    - src/core_llm/checkpoints/domain_adapted_model.pt

Checkpoint format (superset of pretrain format):
    {
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "model_config": dict,
        "tokenizer_path": str,
        "epoch": int,
        "global_step": int,
        "train_loss": float,
        "validation_loss": float,
        "validation_perplexity": float,
        "seed": int,
        "stage": "domain_adapt",
        "domain": str,
        "base_checkpoint": str,
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
DOMAIN_CORPUS_PATH = REPO_ROOT / "data" / "raw" / "clean" / "domain_corpus.txt"
BASE_CHECKPOINT_PATH = REPO_ROOT / "src" / "core_llm" / "checkpoints" / "tiny_model.pt"
DOMAIN_CHECKPOINT_PATH = REPO_ROOT / "src" / "core_llm" / "checkpoints" / "domain_adapted_model.pt"
TOKENIZER_PATH = REPO_ROOT / "src" / "core_llm" / "tokenizer" / "tokenizer.model"

# ── Hyperparameters ───────────────────────────────────────────────────────────
DOMAIN = "deep_learning"
SEED = 42
BATCH_SIZE = 4
CONTEXT_LENGTH = 256   # Must match base checkpoint context_length
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 5
VALIDATION_RATIO = 0.1
PATIENCE = 2
MAX_GRAD_NORM = 1.0


def load_base_checkpoint(checkpoint_path: Path, tokenizer, device):
    """
    Load the base pretrained checkpoint and reconstruct the model.

    Reads model_config from checkpoint so architecture matches exactly.
    Raises RuntimeError with actionable message if checkpoint is missing.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"\nBLOCKED: Base checkpoint not found: {checkpoint_path}\n"
            "Run pretraining first:\n"
            "  python -m src.core_llm.training.pretrain"
        )

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if "model_config" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} is missing 'model_config'. "
            "It may have been saved by an older version. Re-run pretraining."
        )

    config_dict = checkpoint["model_config"]
    # Handle both dict (new format) and ModelConfig object (old format)
    if isinstance(config_dict, dict):
        config = ModelConfig.from_dict(config_dict)
    else:
        config = config_dict

    # Ensure vocabulary size matches the loaded tokenizer
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Checkpoint vocab_size ({config.vocab_size}) != "
            f"tokenizer vocab_size ({tokenizer.vocab_size}). "
            "Ensure the same tokenizer was used for pretraining."
        )

    model = CausalTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, config


def main():
    torch.manual_seed(SEED)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Domain:        {DOMAIN}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = BPETokenizer(str(TOKENIZER_PATH))
    print(f"Vocabulary:    {tokenizer.vocab_size}")

    # ── Domain corpus ─────────────────────────────────────────────────────
    if not DOMAIN_CORPUS_PATH.is_file():
        raise FileNotFoundError(
            f"\nBLOCKED: Domain corpus not found: {DOMAIN_CORPUS_PATH}\n"
            "Supply a UTF-8 text file of domain-specific text.\n"
            "Example domain: deep_learning papers, textbook chapters."
        )

    token_ids = load_corpus_tokens(str(DOMAIN_CORPUS_PATH), tokenizer)
    print(f"Domain tokens: {len(token_ids):,}")

    if len(token_ids) <= CONTEXT_LENGTH:
        raise ValueError(
            f"Domain corpus has only {len(token_ids)} tokens but "
            f"context_length={CONTEXT_LENGTH}. Provide more domain text."
        )

    # ── Dataset split ─────────────────────────────────────────────────────
    split_index = int(len(token_ids) * (1 - VALIDATION_RATIO))
    train_tokens = token_ids[:split_index]
    val_tokens = token_ids[split_index:]

    train_dataset = LanguageModelDataset(train_tokens, CONTEXT_LENGTH)
    val_dataset = LanguageModelDataset(val_tokens, CONTEXT_LENGTH)

    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")

    # ── Load base model ───────────────────────────────────────────────────
    model, config = load_base_checkpoint(BASE_CHECKPOINT_PATH, tokenizer, device)
    print(f"Loaded base checkpoint: {BASE_CHECKPOINT_PATH}")
    print(f"Model parameters:       {count_parameters(model):,}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    DOMAIN_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            out = model(input_ids, target_ids)
            loss = out["loss"]
            if torch.isnan(loss):
                raise RuntimeError("NaN loss during domain adaptation.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1
            global_step += 1

        train_loss = total_train_loss / n_batches if n_batches > 0 else float("inf")
        val_loss = evaluate(model, val_loader, device)
        val_ppl = perplexity(val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": config.to_dict(),
                    "tokenizer_path": str(TOKENIZER_PATH),
                    "epoch": best_epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "validation_perplexity": val_ppl,
                    "seed": SEED,
                    "stage": "domain_adapt",
                    "domain": DOMAIN,
                    "base_checkpoint": str(BASE_CHECKPOINT_PATH),
                },
                DOMAIN_CHECKPOINT_PATH,
            )
            print(f"  ✓ Best domain model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print(f"\nDomain adaptation complete.")
    print(f"Domain: {DOMAIN}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Best val loss:    {best_val_loss:.4f}")
    print(f"Best val PPL:     {perplexity(best_val_loss):.2f}")
    print(f"Checkpoint:       {DOMAIN_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
