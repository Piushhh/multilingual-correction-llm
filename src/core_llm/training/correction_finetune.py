"""
Correction fine-tuning script.

Fine-tunes the domain-adapted model on correction pairs (incorrect → correct).

Prompt masking:
    Only the correction tokens (answer portion) contribute to loss.
    Prompt tokens use ignore_index=-100.

Usage:
    python -m src.core_llm.training.correction_finetune

Requirements:
    - src/core_llm/checkpoints/domain_adapted_model.pt
    - data/correction/correction_pairs.jsonl
    - src/core_llm/tokenizer/tokenizer.model

Output:
    - src/core_llm/checkpoints/correction_model.pt

NOTE: The correction dataset in this repository contains 18 examples
(10 English + 7 Hindi + 1 code-mixed approximation). This is intentionally
a small demonstration dataset. Training metrics reflect this limitation.
Do NOT interpret good training loss as proof of production-quality correction.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[4]
CORRECTION_DATA_PATH = REPO_ROOT / "data" / "correction" / "correction_pairs.jsonl"
BASE_CHECKPOINT_PATH = REPO_ROOT / "src" / "core_llm" / "checkpoints" / "domain_adapted_model.pt"
CORRECTION_CHECKPOINT_PATH = REPO_ROOT / "src" / "core_llm" / "checkpoints" / "correction_model.pt"
TOKENIZER_PATH = REPO_ROOT / "src" / "core_llm" / "tokenizer" / "tokenizer.model"

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED = 42
CONTEXT_LENGTH = 256   # Loaded from checkpoint; this is the fallback
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS = 15
VALIDATION_RATIO = 0.2
PATIENCE = 4
MAX_GRAD_NORM = 1.0


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_examples(path: Path) -> list[dict]:
    """Load and validate JSONL correction examples."""
    if not path.is_file():
        raise FileNotFoundError(
            f"\nBLOCKED: Correction data not found: {path}\n"
            "Expected fields: incorrect, correct, language, domain"
        )

    examples = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

            required = {"incorrect", "correct"}
            missing = required - set(record.keys())
            if missing:
                raise ValueError(
                    f"Record on line {line_num} is missing fields: {missing}"
                )

            examples.append(record)

    if not examples:
        raise ValueError(f"Correction data file is empty: {path}")

    return examples


class CorrectionDataset(Dataset):
    """
    Dataset for correction fine-tuning.

    Each item encodes: "Correct:\\n<incorrect>\\nAnswer:\\n<correct>"
    with labels masked to -100 on the prompt portion so only the
    correction answer tokens contribute to the loss.
    """

    PROMPT_PREFIX = "Correct:\n"
    ANSWER_PREFIX = "\nAnswer:\n"

    def __init__(self, examples: list[dict], tokenizer: BPETokenizer, context_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]

        prompt = self.PROMPT_PREFIX + example["incorrect"] + self.ANSWER_PREFIX
        correction = example["correct"]

        prompt_ids = self.tokenizer.encode(prompt)
        correction_ids = self.tokenizer.encode(correction)

        # Ensure at least 2 correction tokens are present
        max_total = self.context_length
        available_for_prompt = max_total - len(correction_ids)

        if available_for_prompt < 2:
            # Trim correction to make room for at least a minimal prompt
            half = max_total // 2
            correction_ids = correction_ids[:half]
            available_for_prompt = max_total - len(correction_ids)

        prompt_ids = prompt_ids[:available_for_prompt]
        token_ids = (prompt_ids + correction_ids)[:max_total]

        # Build input/label pair with next-token shift
        input_ids = token_ids[:-1]
        labels = token_ids[1:]

        # Mask out prompt tokens in labels so only correction loss is computed
        prompt_label_length = max(0, len(prompt_ids) - 1)
        masked_labels = [-100] * prompt_label_length + labels[prompt_label_length:]

        # Pad to consistent length for batching
        seq_len = len(input_ids)
        pad_len = (self.context_length - 1) - seq_len
        input_ids = input_ids + [self.tokenizer.PAD_ID] * pad_len
        masked_labels = masked_labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(masked_labels, dtype=torch.long),
        )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(checkpoint_path: Path, tokenizer: BPETokenizer, device):
    """
    Load domain-adapted checkpoint.
    Reads model architecture from checkpoint metadata (no hard-coding).
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"\nBLOCKED: Domain-adapted checkpoint not found: {checkpoint_path}\n"
            "Run domain adaptation first:\n"
            "  python -m src.core_llm.training.domain_adapt"
        )

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if "model_config" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'model_config'. Re-run domain_adapt.py "
            "to produce a properly formatted checkpoint."
        )

    config_dict = checkpoint["model_config"]
    if isinstance(config_dict, dict):
        config = ModelConfig.from_dict(config_dict)
    else:
        config = config_dict

    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Checkpoint vocab_size ({config.vocab_size}) != "
            f"tokenizer vocab_size ({tokenizer.vocab_size})."
        )

    model = CausalTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, config


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss(model, input_ids, labels):
    out = model(input_ids)
    logits = out["logits"]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = compute_loss(model, input_ids, labels)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss during correction fine-tuning.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n if n > 0 else float("inf")


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        total += compute_loss(model, input_ids, labels).item()
        n += 1
    model.train()
    return total / n if n > 0 else float("inf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BPETokenizer(str(TOKENIZER_PATH))
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    examples = load_examples(CORRECTION_DATA_PATH)
    print(f"\nCorrection examples: {len(examples)}")

    english = sum(1 for x in examples if x.get("language") == "en")
    hindi = sum(1 for x in examples if x.get("language") == "hi")
    mixed = sum(1 for x in examples if x.get("language") == "mixed")
    print(f"  English: {english}  Hindi: {hindi}  Code-mixed: {mixed}")
    print(
        "\nNOTE: This dataset is intentionally small (demonstration only). "
        "Training metrics reflect this limitation."
    )

    dataset = CorrectionDataset(examples, tokenizer, CONTEXT_LENGTH)

    val_size = max(1, int(len(dataset) * VALIDATION_RATIO))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples:   {len(train_ds)}")
    print(f"Val samples:     {len(val_ds)}")

    model, config = load_base_model(BASE_CHECKPOINT_PATH, tokenizer, device)
    print(f"Loaded domain checkpoint: {BASE_CHECKPOINT_PATH}")
    print(f"Context length: {config.context_length}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CORRECTION_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        val_ppl = math.exp(min(val_loss, 20.0))

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
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "validation_perplexity": val_ppl,
                    "seed": SEED,
                    "stage": "correction_finetune",
                    "domain": "deep_learning",
                    "base_checkpoint": str(BASE_CHECKPOINT_PATH),
                    "correction_examples": len(examples),
                },
                CORRECTION_CHECKPOINT_PATH,
            )
            print(f"  ✓ Best correction model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print(f"\nCorrection fine-tuning complete.")
    print(f"Best epoch:   {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val PPL:  {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"Checkpoint:   {CORRECTION_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
